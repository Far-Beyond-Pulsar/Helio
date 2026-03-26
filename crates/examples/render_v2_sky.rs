//! Sky example using helio v3.
//!
//! A simple scene with a sun directional light and three colored point lights.
//! Q/E keys rotate the sun, simulating time of day.
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Q/E         — rotate sun left/right (changes time of day)
//!   Mouse drag  — look around (click to grab cursor)
//!   Escape      — release cursor / exit

mod v3_demo_common;

use helio::{
    required_wgpu_features, required_wgpu_limits, Camera, LightId, MeshId, Renderer, RendererConfig,
};
use v3_demo_common::{
    box_mesh, cube_mesh, directional_light, make_material, plane_mesh, point_light,
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

    cube1: MeshId,
    cube2: MeshId,
    cube3: MeshId,
    ground: MeshId,
    roof: MeshId,

    // Free-camera state
    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    // Time-of-day: sun_angle=0 → noon, PI → midnight
    sun_angle: f32,

    // Scene state
    sun_light_id: LightId,
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
                        .with_title("Helio – Volumetric Sky")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .expect("Failed to create window"),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
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

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Main Device"),
            required_features: required_wgpu_features(adapter.features()),
            required_limits: required_wgpu_limits(adapter.limits()),
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
        };
        surface.configure(&device, &config);

        let mut renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            RendererConfig::new(size.width, size.height, surface_format),
        );

        let mat = renderer.scene_mut().insert_material(make_material(
            [0.7, 0.7, 0.72, 1.0],
            0.7,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));

        let cube1 = renderer.scene_mut().insert_mesh(cube_mesh([0.0, 0.0, 0.0], 0.5));
        let cube2 = renderer.scene_mut().insert_mesh(cube_mesh([0.0, 0.0, 0.0], 0.4));
        let cube3 = renderer.scene_mut().insert_mesh(cube_mesh([0.0, 0.0, 0.0], 0.3));
        let ground = renderer.scene_mut().insert_mesh(plane_mesh([0.0, 0.0, 0.0], 20.0));
        let roof = renderer.scene_mut().insert_mesh(box_mesh([0.0, 0.0, 0.0], [4.5, 0.15, 4.5]));

        let _ = v3_demo_common::insert_object(&mut renderer, cube1, mat, glam::Mat4::from_translation(glam::Vec3::new( 0.0,  0.5,  0.0)), 0.5);
        let _ = v3_demo_common::insert_object(&mut renderer, cube2, mat, glam::Mat4::from_translation(glam::Vec3::new(-2.0,  0.4, -1.0)), 0.4);
        let _ = v3_demo_common::insert_object(&mut renderer, cube3, mat, glam::Mat4::from_translation(glam::Vec3::new( 2.0,  0.3,  0.5)), 0.3);
        let _ =
            v3_demo_common::insert_object(&mut renderer, ground, mat, glam::Mat4::IDENTITY, 20.0);
        let _ = v3_demo_common::insert_object(&mut renderer, roof, mat, glam::Mat4::from_translation(glam::Vec3::new(0.0, 2.85, 0.0)), 4.5);

        // Compute initial sun direction from starting sun_angle=1.0
        let init_sun_dir = glam::Vec3::new(1.0_f32.cos() * 0.3, 1.0_f32.sin(), 0.5).normalize();
        let init_light_dir = [-init_sun_dir.x, -init_sun_dir.y, -init_sun_dir.z];
        let init_elev = init_sun_dir.y.clamp(-1.0, 1.0);
        let init_lux = (init_elev * 3.0).clamp(0.0, 1.0);
        let sun_light_id = renderer.scene_mut().insert_light(directional_light(
            init_light_dir,
            [1.0, 0.85, 0.7],
            (init_lux * 0.35).max(0.01),
        ));
        renderer.scene_mut().insert_light(point_light([0.0, 2.5, 0.0], [1.0, 0.85, 0.6], 4.0, 8.0));
        renderer.scene_mut().insert_light(point_light([-2.5, 2.0, -1.5], [0.4, 0.6, 1.0], 3.5, 7.0));
        renderer.scene_mut().insert_light(point_light([2.5, 1.8, 1.5], [1.0, 0.3, 0.3], 3.0, 6.0));
        renderer.set_ambient([0.15, 0.18, 0.25], 0.08);

        renderer.scene_mut().set_volumetric_clouds(Some(helio::VolumetricClouds {
            coverage: 0.7,
            density: 0.8,
            base: 1200.0,
            top: 1800.0,
            wind_x: 0.8,
            wind_z: 0.2,
            speed: 1.3,
            skylight_intensity: 0.25,
        }));

        self.state = Some(AppState {
            window,
            surface,
            device,
            surface_format,
            renderer,
            last_frame: std::time::Instant::now(),
            cube1,
            cube2,
            cube3,
            ground,
            roof,
            cam_pos: glam::Vec3::new(0.0, 2.5, 7.0),
            cam_yaw: 0.0,
            cam_pitch: -0.2,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            // Start at a nice afternoon angle (sun ~50° above horizon)
            sun_angle: 1.0,
            sun_light_id,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };

        match event {
            WindowEvent::CloseRequested => {
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
                let cfg = wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: state.surface_format,
                    width: size.width,
                    height: size.height,
                    present_mode: wgpu::PresentMode::Fifo,
                    alpha_mode: wgpu::CompositeAlphaMode::Auto,
                    view_formats: vec![],
                    desired_maximum_frame_latency: 2,
                };
                state.surface.configure(&state.device, &cfg);
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
        if let Some(s) = &self.state {
            s.window.request_redraw();
        }
    }
}

impl AppState {
    fn render(&mut self, dt: f32) {
        const SPEED: f32 = 5.0;
        const LOOK_SENS: f32 = 0.002;
        const SUN_SPEED: f32 = 0.5; // radians/sec

        // Sun rotation (Q/E keys)
        if self.keys.contains(&KeyCode::KeyQ) {
            self.sun_angle -= SUN_SPEED * dt;
        }
        if self.keys.contains(&KeyCode::KeyE) {
            self.sun_angle += SUN_SPEED * dt;
        }

        // Camera look
        self.cam_yaw += self.mouse_delta.0 * LOOK_SENS;
        self.cam_pitch = (self.cam_pitch - self.mouse_delta.1 * LOOK_SENS).clamp(-1.5, 1.5);
        self.mouse_delta = (0.0, 0.0);

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let forward = glam::Vec3::new(sy * cp, sp, -cy * cp);
        let right = glam::Vec3::new(cy, 0.0, sy);

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
            self.cam_pos += glam::Vec3::Y * SPEED * dt;
        }
        if self.keys.contains(&KeyCode::ShiftLeft) {
            self.cam_pos -= glam::Vec3::Y * SPEED * dt;
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
            1000.0,
        );

        // Sun direction: orbits in the XY plane (rotate sun_angle around Z axis)
        let sun_dir =
            glam::Vec3::new(self.sun_angle.cos() * 0.3, self.sun_angle.sin(), 0.5).normalize();
        // SceneLight direction = "ray direction" (toward scene), so negate the toward-sun vector
        let light_dir = [-sun_dir.x, -sun_dir.y, -sun_dir.z];

        // Sun intensity dims at horizon/night
        let sun_elev = sun_dir.y.clamp(-1.0, 1.0);
        let sun_lux = (sun_elev * 3.0).clamp(0.0, 1.0);
        let sun_color = [
            1.0_f32.min(1.0 + (1.0 - sun_elev) * 0.3), // warmer at horizon
            (0.85 + sun_elev * 0.15).clamp(0.0, 1.0),
            (0.7 + sun_elev * 0.3).clamp(0.0, 1.0),
        ];

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => {
                log::warn!("Surface error: {:?}", e);
                return;
            }
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Update dynamic sun light
        let _ = self.renderer.scene_mut().update_light(
            self.sun_light_id,
            directional_light(light_dir, sun_color, (sun_lux * 0.35).max(0.01)),
        );
        if let Err(e) = self.renderer.render(&camera, &view) {
            log::error!("Render error: {:?}", e);
        }

        output.present();
    }
}

