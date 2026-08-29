//! HDR Display Output Demo.
//!
//! Demonstrates HDR10 (PQ ST 2084), scRGB, and Passthrough output modes
//! with a scene containing emissive objects at extreme brightness levels.
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Mouse drag  — look around (click to grab cursor)
//!   H           — cycle HDR output mode
//!   Escape      — release cursor / exit

mod v3_demo_common;

use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera,
    DebugDrawState, HdrOutputMode, Renderer, RendererConfig, Scene,
};
use helio_default_graphs::build_default_graph;
use v3_demo_common::{cube_mesh, directional_light, make_material, plane_mesh, point_light};

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use std::collections::HashSet;
use std::sync::Arc;

fn main() {
    env_logger::init();
    log::info!("Starting Helio HDR Demo");

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
    queue: Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
    renderer: Renderer,
    last_frame: std::time::Instant,
    start_time: std::time::Instant,

    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    hdr_mode: u32,
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
                        .with_title("Helio — HDR Display Output")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .expect("Failed to create window"),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("Failed to find adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Main Device"),
            required_features: required_wgpu_features(adapter.features()),
            required_limits: required_wgpu_limits(adapter.limits()),
            experimental_features: required_experimental_features(adapter.features()),
            ..Default::default()
        }))
        .expect("Failed to create device");

        device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| {
            panic!("[GPU UNCAPTURED ERROR] {:?}", e);
        }));
        let info = adapter.get_info();
        println!(
            "[WGPU] Backend: {:?}, Device: {}, Driver: {}",
            info.backend, info.name, info.driver
        );
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
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
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&device, &config);

        let config = RendererConfig::new(size.width, size.height, surface_format);
        let scene = Scene::new(device.clone(), queue.clone());
        let debug_camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Camera Buffer"),
            size: std::mem::size_of::<helio::DebugCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cull_stats_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cull Stats Buffer"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let debug_state = Arc::new(std::sync::Mutex::new(DebugDrawState::default()));
        let graph = build_default_graph(
            &device,
            &queue,
            &scene,
            config,
            debug_state.clone(),
            &debug_camera_buf,
            &cull_stats_buf,
            None,
        );
        let mut renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            config.surface_format,
            config.width,
            config.height,
            config.render_scale,
            config,
            scene,
            graph,
            debug_state,
            debug_camera_buf,
            cull_stats_buf,
        );
        renderer.set_editor_mode(true);

        let white = renderer.scene_mut().insert_material(make_material(
            [0.9, 0.9, 0.92, 1.0],
            0.6,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));
        let emissive_red = renderer.scene_mut().insert_material(make_material(
            [1.0, 0.1, 0.1, 1.0],
            0.3,
            0.0,
            [10.0, 0.5, 0.5],
            10.0,
        ));
        let emissive_green = renderer.scene_mut().insert_material(make_material(
            [0.1, 1.0, 0.1, 1.0],
            0.3,
            0.0,
            [0.5, 10.0, 0.5],
            10.0,
        ));
        let emissive_blue = renderer.scene_mut().insert_material(make_material(
            [0.1, 0.1, 1.0, 1.0],
            0.3,
            0.0,
            [0.5, 0.5, 10.0],
            10.0,
        ));
        let emissive_sun = renderer.scene_mut().insert_material(make_material(
            [1.0, 0.9, 0.7, 1.0],
            0.2,
            0.0,
            [50.0, 45.0, 35.0],
            50.0,
        ));
        let metal = renderer.scene_mut().insert_material(make_material(
            [0.95, 0.93, 0.88, 1.0],
            0.1,
            1.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));

        let ground = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(plane_mesh([0.0, 0.0, 0.0], 8.0)))
            .as_mesh()
            .unwrap();
        let _ =
            v3_demo_common::insert_object(&mut renderer, ground, white, glam::Mat4::IDENTITY, 8.0);

        let red_cube = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(cube_mesh([-1.5, 0.5, -1.0], 0.5)))
            .as_mesh()
            .unwrap();
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            red_cube,
            emissive_red,
            glam::Mat4::IDENTITY,
            0.5,
        );

        let green_cube = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(cube_mesh([1.5, 0.5, -1.0], 0.5)))
            .as_mesh()
            .unwrap();
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            green_cube,
            emissive_green,
            glam::Mat4::IDENTITY,
            0.5,
        );

        let blue_cube = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(cube_mesh([0.0, 0.5, 1.5], 0.5)))
            .as_mesh()
            .unwrap();
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            blue_cube,
            emissive_blue,
            glam::Mat4::IDENTITY,
            0.5,
        );

        let metal_cube = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(cube_mesh([-1.5, 0.5, 2.5], 0.5)))
            .as_mesh()
            .unwrap();
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            metal_cube,
            metal,
            glam::Mat4::IDENTITY,
            0.5,
        );

        let sun_sphere = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(cube_mesh([3.0, 4.0, -3.0], 0.4)))
            .as_mesh()
            .unwrap();
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            sun_sphere,
            emissive_sun,
            glam::Mat4::IDENTITY,
            0.4,
        );

        renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::light(directional_light(
                [0.3, -0.8, 0.5],
                [1.0, 0.95, 0.85],
                15.0,
            )));
        renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::light(point_light(
                [3.0, 4.0, -3.0],
                [1.0, 0.9, 0.7],
                20.0,
                15.0,
            )));

        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_format,
            renderer,
            last_frame: std::time::Instant::now(),
            start_time: std::time::Instant::now(),
            cam_pos: glam::Vec3::new(0.0, 3.0, 6.0),
            cam_yaw: 0.0,
            cam_pitch: -0.3,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            hdr_mode: 0,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let state = match self.state.as_mut() {
            Some(s) => s,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        state: ElementState::Released,
                        ..
                    },
                ..
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
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => {
                if key_state == ElementState::Pressed {
                    state.keys.insert(code);
                    if code == KeyCode::KeyH {
                        state.hdr_mode = (state.hdr_mode + 1) % 4;
                        let name = match state.hdr_mode {
                            0 => "LDR (sRGB)",
                            1 => "HDR10 (PQ)",
                            2 => "scRGB",
                            _ => "Passthrough",
                        };
                        log::info!("[HDR Demo] Switched to {}", name);
                        state.window.set_title(&format!("Helio — HDR: {}", name));
                    }
                } else {
                    state.keys.remove(&code);
                }
            }
            WindowEvent::CursorMoved {
                device_id: _,
                position,
            } => {
                if state.cursor_grabbed {
                    let center = state.window.inner_size();
                    let dx = position.x as f32 - center.width as f32 / 2.0;
                    let dy = position.y as f32 - center.height as f32 / 2.0;
                    state.mouse_delta = (dx, dy);
                    let _ = state
                        .window
                        .set_cursor_position(winit::dpi::PhysicalPosition::new(
                            center.width as f64 / 2.0,
                            center.height as f64 / 2.0,
                        ));
                }
            }
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state: ElementState::Pressed,
                ..
            } => {
                state.cursor_grabbed = true;
                let _ = state.window.set_cursor_grab(CursorGrabMode::Confined);
                state.window.set_cursor_visible(false);
            }
            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    let config = wgpu::SurfaceConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        format: state.surface_format,
                        width: size.width,
                        height: size.height,
                        present_mode: wgpu::PresentMode::Fifo,
                        alpha_mode: wgpu::CompositeAlphaMode::Auto,
                        view_formats: vec![],
                        desired_maximum_frame_latency: 2,
                        color_space: wgpu::SurfaceColorSpace::Auto,
                    };
                    state.surface.configure(&state.device, &config);
                    state.renderer.set_render_size(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt = (now - state.last_frame).as_secs_f32().min(0.05);
                state.last_frame = now;
                let time = state.start_time.elapsed().as_secs_f32();

                let speed = 5.0;
                let (sy, cy) = state.cam_yaw.sin_cos();
                let (sp, cp) = state.cam_pitch.sin_cos();
                let forward = glam::Vec3::new(sy * cp, sp, -cy * cp);
                let right = glam::Vec3::new(cy, 0.0, sy);

                if state.keys.contains(&KeyCode::KeyW) {
                    state.cam_pos += forward * speed * dt;
                }
                if state.keys.contains(&KeyCode::KeyS) {
                    state.cam_pos -= forward * speed * dt;
                }
                if state.keys.contains(&KeyCode::KeyA) {
                    state.cam_pos -= right * speed * dt;
                }
                if state.keys.contains(&KeyCode::KeyD) {
                    state.cam_pos += right * speed * dt;
                }
                if state.keys.contains(&KeyCode::Space) {
                    state.cam_pos.y += speed * dt;
                }
                if state.keys.contains(&KeyCode::ShiftLeft) {
                    state.cam_pos.y -= speed * dt;
                }

                state.cam_yaw -= state.mouse_delta.0 * 0.005;
                state.cam_pitch =
                    (state.cam_pitch - state.mouse_delta.1 * 0.005).clamp(-1.55, 1.55);
                state.mouse_delta = (0.0, 0.0);

                let aspect = state.window.inner_size().width as f32
                    / state.window.inner_size().height.max(1) as f32;

                let mut camera = Camera::perspective_look_at(
                    state.cam_pos,
                    state.cam_pos + forward,
                    glam::Vec3::Y,
                    std::f32::consts::FRAC_PI_4,
                    aspect,
                    0.1,
                    200.0,
                );
                match state.hdr_mode {
                    0 => {
                        camera.postprocess_settings.hdr_output_mode = HdrOutputMode::Ldr;
                        camera.postprocess_settings.tonemap_operator = helio::TonemapOperator::Aces;
                    }
                    1 => {
                        camera.postprocess_settings.hdr_output_mode = HdrOutputMode::Hdr10;
                        camera.postprocess_settings.tonemap_operator = helio::TonemapOperator::None;
                    }
                    2 => {
                        camera.postprocess_settings.hdr_output_mode = HdrOutputMode::ScRgb;
                        camera.postprocess_settings.tonemap_operator = helio::TonemapOperator::None;
                    }
                    _ => {
                        camera.postprocess_settings.hdr_output_mode = HdrOutputMode::Passthrough;
                        camera.postprocess_settings.tonemap_operator = helio::TonemapOperator::None;
                    }
                }

                let output = match state.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(texture)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
                    _ => return,
                };
                let view = output
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                if let Err(e) = state.renderer.render(&camera, &view) {
                    log::error!("Render error: {:?}", e);
                }
                state.queue.present(output);

                state.window.request_redraw();
            }
            _ => {}
        }
    }
}
