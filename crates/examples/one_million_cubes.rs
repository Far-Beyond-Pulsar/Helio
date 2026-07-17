//! 1 Million Cubes — GPU-driven instancing stress test.
//!
//! Renders 1,000,000 cubes in a 100×100×100 grid, all sharing a single mesh
//! and one of 10 materials.  The renderer automatically batches objects with
//! the same (mesh, material) pair into instanced indirect draw calls, so these
//! 1M cubes become at most 10 GPU draw calls.
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Mouse drag  — look around (click to grab cursor)
//!   Escape      — release cursor / exit

mod v3_demo_common;

use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera,
    DebugDrawState, Renderer, RendererConfig, Scene,
};
use helio_default_graphs::build_default_graph;
use v3_demo_common::{
    cube_mesh, directional_light, insert_object_with_movability, make_material, point_light,
};

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
    log::info!("Starting 1 Million Cubes Instancing Demo");

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
    start_time: std::time::Instant,

    cam_pos: glam::Vec3,
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

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Helio – 1 Million Cubes")
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
            apply_limit_buckets: true,
        }))
        .expect("Failed to find adapter");

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
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
        let surface_config = wgpu::SurfaceConfiguration {
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
        surface.configure(&device, &surface_config);

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

        let palette = [
            [0.91, 0.18, 0.18, 1.0],
            [0.18, 0.91, 0.18, 1.0],
            [0.18, 0.18, 0.91, 1.0],
            [0.91, 0.91, 0.18, 1.0],
            [0.91, 0.18, 0.91, 1.0],
            [0.18, 0.91, 0.91, 1.0],
            [0.91, 0.55, 0.18, 1.0],
            [0.55, 0.18, 0.91, 1.0],
            [0.91, 0.55, 0.55, 1.0],
            [0.55, 0.91, 0.55, 1.0],
        ];

        let materials: Vec<_> = palette
            .iter()
            .map(|&color| {
                renderer
                    .scene_mut()
                    .insert_material(make_material(color, 0.5, 0.1, [0.0, 0.0, 0.0], 0.0))
            })
            .collect();

        let cube_mesh_id = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(cube_mesh([0.0, 0.0, 0.0], 0.4)))
            .as_mesh()
            .unwrap();

        // 100×100×100 = 1,000,000 cubes in a centred grid
        let grid_size = 100i32;
        let spacing = 1.5;
        let half = grid_size as f32 * spacing * 0.5;

        let timer = std::time::Instant::now();
        let mat_count = materials.len();

        for x in 0..grid_size {
            for y in 0..grid_size {
                for z in 0..grid_size {
                    let pos = glam::Vec3::new(
                        x as f32 * spacing - half,
                        y as f32 * spacing - half,
                        z as f32 * spacing - half,
                    );

                    let mat_idx = (((y * grid_size + z) as f32 / (grid_size * grid_size) as f32)
                        * mat_count as f32) as usize
                        % mat_count;

                    let transform = glam::Mat4::from_translation(pos);
                    let _ = insert_object_with_movability(
                        &mut renderer,
                        cube_mesh_id,
                        materials[mat_idx],
                        transform,
                        0.5,
                        Some(helio::Movability::Static),
                    );
                }
            }
        }

        let elapsed = timer.elapsed();
        println!(
            "Created 1,000,000 cube instances in {:.2}s",
            elapsed.as_secs_f32()
        );

        renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::light(directional_light(
                [0.5, -0.8, 0.3],
                [1.0, 0.95, 0.85],
                8.0,
            )));

        renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::light(point_light(
                [-60.0, 40.0, -60.0],
                [0.3, 0.6, 1.0],
                4.0,
                120.0,
            )));

        renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::light(point_light(
                [60.0, 40.0, 60.0],
                [1.0, 0.6, 0.3],
                4.0,
                120.0,
            )));

        self.state = Some(AppState {
            window,
            surface,
            device,
            surface_format,
            renderer,
            last_frame: std::time::Instant::now(),
            start_time: std::time::Instant::now(),
            cam_pos: glam::Vec3::new(0.0, 50.0, 120.0),
            cam_yaw: 0.0,
            cam_pitch: -0.35,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                log::info!("Shutting down");
                event_loop.exit();
            }

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
                    let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                    state.window.set_cursor_visible(true);
                } else {
                    event_loop.exit();
                }
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ks,
                        physical_key: PhysicalKey::Code(key),
                        ..
                    },
                ..
            } => match ks {
                ElementState::Pressed => {
                    state.keys.insert(key);
                }
                ElementState::Released => {
                    state.keys.remove(&key);
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
                        state.window.set_cursor_visible(false);
                        state.cursor_grabbed = true;
                    }
                }
            }

            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                let sc = wgpu::SurfaceConfiguration {
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
                state.surface.configure(&state.device, &sc);
                state.renderer.set_render_size(size.width, size.height);
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

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
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

impl AppState {
    fn render(&mut self, dt: f32) {
        const SPEED: f32 = 20.0;
        const LOOK_SENS: f32 = 0.002;

        self.cam_yaw += self.mouse_delta.0 * LOOK_SENS;
        self.cam_pitch = (self.cam_pitch - self.mouse_delta.1 * LOOK_SENS).clamp(-1.5, 1.5);
        self.mouse_delta = (0.0, 0.0);

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let forward = glam::Vec3::new(sy * cp, sp, -cy * cp);
        let right = glam::Vec3::new(cy, 0.0, sy);
        let up = glam::Vec3::Y;

        if self.keys.contains(&KeyCode::KeyW) {
            self.cam_pos += forward * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::KeyS) {
            self.cam_pos -= forward * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::KeyA) {
            self.cam_pos -= right * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::KeyD) {
            self.cam_pos += right * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::Space) {
            self.cam_pos += up * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::ShiftLeft) {
            self.cam_pos -= up * SPEED * dt;
        }

        let size = self.window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;

        let camera = Camera::perspective_look_at(
            self.cam_pos,
            self.cam_pos + forward,
            glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            aspect,
            0.1,
            500.0,
        );

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) => t,
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => {
                log::warn!("surface acquire failed");
                return;
            }
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        if let Err(e) = self.renderer.render(&camera, &view) {
            log::error!("Render error: {:?}", e);
        }

        self.renderer.queue().present(output);
    }
}
