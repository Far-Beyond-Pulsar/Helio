//! Lens Flare Demo — a bright directional light with lens flare enabled.
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Mouse drag  — look around (click to grab cursor)
//!   Escape      — release cursor / exit

mod v3_demo_common;

use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera,
    DebugDrawState, GpuLight, LightType, Renderer, RendererConfig, Scene,
};
use helio_default_graphs::build_default_graph;
use v3_demo_common::{box_mesh, make_material, plane_mesh};

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

    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    velocity: glam::Vec3,
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
                        .with_title("Helio — Lens Flare Demo")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .expect("window"),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_features: required_wgpu_features(adapter.features()),
            required_limits: required_wgpu_limits(adapter.limits()),
            experimental_features: required_experimental_features(adapter.features()),
            ..Default::default()
        }))
        .expect("device");

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

        let caps = surface.get_capabilities(&adapter);
        let format = caps
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
                format,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
                color_space: wgpu::SurfaceColorSpace::Auto,
            },
        );

        let config = RendererConfig::new(size.width, size.height, format);
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

        // ── Scene objects ──────────────────────────────────────────────────────
        let mat_wall = renderer.scene_mut().insert_material(make_material(
            [0.6, 0.58, 0.55, 1.0],
            0.7,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));
        let mat_floor = renderer.scene_mut().insert_material(make_material(
            [0.3, 0.28, 0.25, 1.0],
            0.4,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));
        // Ground plane
        let floor = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(plane_mesh([0.0, -0.5, 0.0], 6.0)))
            .as_mesh()
            .unwrap();
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            floor,
            mat_floor,
            glam::Mat4::IDENTITY,
            6.0,
        );

        // Back wall — catches the light
        let back_wall = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(box_mesh(
                [0.0, 1.5, -5.0],
                [6.0, 3.0, 0.1],
            )))
            .as_mesh()
            .unwrap();
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            back_wall,
            mat_wall,
            glam::Mat4::IDENTITY,
            6.0,
        );

        // Some pillars / columns to create depth
        for (x, z) in &[(-2.5, -2.0), (2.5, -2.0), (-2.5, 2.0), (2.5, 2.0)] {
            let pillar = renderer
                .scene_mut()
                .insert_actor(helio::SceneActor::mesh(box_mesh(
                    [*x, 0.5, *z],
                    [0.3, 1.5, 0.3],
                )))
                .as_mesh()
                .unwrap();
            let _ = v3_demo_common::insert_object(
                &mut renderer,
                pillar,
                mat_wall,
                glam::Mat4::IDENTITY,
                1.0,
            );
        }

        // ── Lighting ───────────────────────────────────────────────────────────
        // Bright directional light shining toward the scene from above-right-front
        let sun_dir = glam::Vec3::new(-0.4, -0.6, 0.7).normalize();
        renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::light(GpuLight {
                position_range: [0.0, 0.0, 0.0, f32::MAX],
                direction_outer: [sun_dir.x, sun_dir.y, sun_dir.z, 0.0],
                color_intensity: [1.0, 0.95, 0.85, 6.0],
                shadow_index: 0,
                light_type: LightType::Directional as u32,
                inner_angle: 0.0,
                _pad: 0,
                god_rays_enabled: 0,
                god_rays_density: 1.0,
                god_rays_weight: 0.6,
                god_rays_decay: 1.0,
                god_rays_exposure: 0.7,
                flare_enabled: 0,
                flare_type: 0,
                flare_intensity: 0.0,
                flare_scale: 0.0,
                flare_tint_r: 0.0,
                flare_tint_g: 0.0,
                flare_tint_b: 0.0,
                ies_profile_index: 0,
                light_function_index: 0,
                ies_angle_scale: 0.0,
                ies_angle_offset: 0.0,
            }));

        // A few fill point lights
        // Bright point light with lens flare — placed off-centre so the ghosts spread diagonally
        renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::light(GpuLight {
                position_range: [-1.5, 2.0, -1.0, 8.0],
                direction_outer: [0.0, -1.0, 0.0, 0.0],
                color_intensity: [1.0, 0.85, 0.55, 8.0],
                shadow_index: u32::MAX,
                light_type: LightType::Point as u32,
                inner_angle: 0.0,
                _pad: 0,
                god_rays_enabled: 0,
                god_rays_density: 1.0,
                god_rays_weight: 0.6,
                god_rays_decay: 1.0,
                god_rays_exposure: 0.7,
                flare_enabled: 1,
                flare_type: 1,
                flare_intensity: 0.25,
                flare_scale: 1.0,
                flare_tint_r: 1.0,
                flare_tint_g: 0.7,
                flare_tint_b: 0.35,
                ies_profile_index: 0,
                light_function_index: 0,
                ies_angle_scale: 0.0,
                ies_angle_offset: 0.0,
            }));

        renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::light(GpuLight {
                position_range: [-3.0, 1.0, -3.0, 5.0],
                direction_outer: [0.0, -1.0, 0.0, 0.0],
                color_intensity: [0.3, 0.4, 0.6, 2.0],
                shadow_index: 0,
                light_type: LightType::Point as u32,
                inner_angle: 0.0,
                _pad: 0,
                ..Default::default()
            }));
        renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::light(GpuLight {
                position_range: [3.0, 1.0, 2.0, 4.0],
                direction_outer: [0.0, -1.0, 0.0, 0.0],
                color_intensity: [0.6, 0.3, 0.2, 1.5],
                shadow_index: 0,
                light_type: LightType::Point as u32,
                inner_angle: 0.0,
                _pad: 0,
                ..Default::default()
            }));

        renderer.set_ambient([0.05, 0.05, 0.08], 0.02);
        renderer.set_clear_color([0.01, 0.01, 0.03, 1.0]);

        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_format: format,
            renderer,
            last_frame: std::time::Instant::now(),
            cam_pos: glam::Vec3::new(0.0, 1.2, 3.5),
            cam_yaw: 0.0,
            cam_pitch: -0.1,
            velocity: glam::Vec3::ZERO,
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
                        color_space: wgpu::SurfaceColorSpace::Auto,
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
                    let ok = state
                        .window
                        .set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                        .is_ok();
                    if ok {
                        state.cursor_grabbed = true;
                        state.window.set_cursor_visible(false);
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt = now.duration_since(state.last_frame).as_secs_f32().min(0.05);
                state.last_frame = now;
                state.update(dt);

                let size = state.window.inner_size();
                let camera = state.camera(size.width, size.height);

                let output = match state.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(texture)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
                    _ => return,
                };
                let view = output
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                if let Err(e) = state.renderer.render(&camera, &view) {
                    log::error!("render error: {:?}", e);
                }
                state.queue.present(output);
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

// ── fly-camera helpers ─────────────────────────────────────────────────────────

const LOOK_SENS: f32 = 0.002;
const FLY_SPEED: f32 = 3.0;
const DRAG: f32 = 8.0;

impl AppState {
    fn update(&mut self, dt: f32) {
        let (dx, dy) = self.mouse_delta;
        self.mouse_delta = (0.0, 0.0);
        self.cam_yaw -= dx * LOOK_SENS;
        self.cam_pitch = (self.cam_pitch - dy * LOOK_SENS).clamp(-1.5, 1.5);

        let orientation =
            glam::Quat::from_euler(glam::EulerRot::YXZ, self.cam_yaw, self.cam_pitch, 0.0);
        let forward = orientation * -glam::Vec3::Z;
        let right = orientation * glam::Vec3::X;
        let up = glam::Vec3::Y;

        let mut accel = glam::Vec3::ZERO;
        if self.keys.contains(&KeyCode::KeyW) {
            accel += forward;
        }
        if self.keys.contains(&KeyCode::KeyS) {
            accel -= forward;
        }
        if self.keys.contains(&KeyCode::KeyA) {
            accel -= right;
        }
        if self.keys.contains(&KeyCode::KeyD) {
            accel += right;
        }
        if self.keys.contains(&KeyCode::Space) {
            accel += up;
        }
        if self.keys.contains(&KeyCode::ShiftLeft) {
            accel -= up;
        }
        if accel.length_squared() > 0.0 {
            accel = accel.normalize();
        }

        self.velocity += accel * FLY_SPEED * dt;
        self.velocity /= 1.0 + DRAG * dt;
        self.cam_pos += self.velocity * dt;
    }

    fn camera(&self, width: u32, height: u32) -> Camera {
        let orientation =
            glam::Quat::from_euler(glam::EulerRot::YXZ, self.cam_yaw, self.cam_pitch, 0.0);
        let target = self.cam_pos + orientation * -glam::Vec3::Z;
        let up = orientation * glam::Vec3::Y;
        Camera::perspective_look_at(
            self.cam_pos,
            target,
            up,
            std::f32::consts::FRAC_PI_4,
            width as f32 / height.max(1) as f32,
            0.01,
            100.0,
        )
    }
}
