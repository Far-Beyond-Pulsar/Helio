//! Depth of Field demo — shows bokeh DOF with objects at varying distances.
//!
//! A long table of coloured spheres and boxes at different depths.
//! The camera faces down the Z axis with DOF enabled. Use mouse to look,
//! WASD to move, and number keys to adjust DOF parameters in real-time.
//!
//! Controls:
//!   WASD/Space/Shift  move              mouse drag  look (click to grab)
//!   Tab               toggle DOF on/off
//!   Scroll Up/Down    rack focus (focal distance)
//!   1 / 2             focal region +/- (range considered "in focus")
//!   3 / 4             aperture blades (3-11, 0 = Gaussian)
//!   5 / 6             max bokeh size +/- (blur radius in pixels)
//!   7 / 8             sensor diagonal +/- (affects CoC scale)
//!   9 / 0             aperture rotation +/- (radians)

mod v3_demo_common;

use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera,
    DebugDrawState, Renderer, RendererConfig, Scene,
};
use helio_default_graphs::build_default_graph;
use v3_demo_common::{
    box_mesh, cube_mesh, directional_light, insert_object, make_material, plane_mesh, sphere_mesh,
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

const SCENE_HALF_Z: f32 = 28.0;
const TABLE_Y: f32 = 1.0;

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
    surface_format: wgpu::TextureFormat,
    renderer: Renderer,
    last_frame: std::time::Instant,

    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    just_pressed: Vec<KeyCode>,
    mouse_delta: (f32, f32),
    cursor_grabbed: bool,
    scroll_delta: f32,

    // DOF state
    dof_enabled: bool,
    dof_focal_distance: f32,
    dof_focal_region: f32,
    dof_aperture_blades: u32,
    dof_max_bokeh_size: f32,
    dof_aperture_rotation: f32,
    dof_sensor_diagonal: f32,
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
                        .with_title("Helio – Depth of Field Demo")
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

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Main Device"),
            required_features: required_wgpu_features(adapter.features()),
            required_limits: required_wgpu_limits(adapter.limits()),
            experimental_features: required_experimental_features(adapter.features()),
            ..Default::default()
        }))
        .expect("Failed to create device");

        device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| {
            log::error!("wgpu uncaptured error: {:?}", e);
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
        let cfg = wgpu::SurfaceConfiguration {
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
        surface.configure(&device, &cfg);

        let config =
            RendererConfig::new(size.width, size.height, surface_format).with_render_scale(1.0);
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

        // ── Materials ───────────────────────────────────────────────────

        let red_mat = renderer.scene_mut().insert_material(make_material(
            [0.9, 0.15, 0.15, 1.0],
            0.4,
            0.6,
            [0.0, 0.0, 0.0],
            0.0,
        ));
        let green_mat = renderer.scene_mut().insert_material(make_material(
            [0.15, 0.9, 0.15, 1.0],
            0.4,
            0.6,
            [0.0, 0.0, 0.0],
            0.0,
        ));
        let blue_mat = renderer.scene_mut().insert_material(make_material(
            [0.15, 0.3, 0.9, 1.0],
            0.4,
            0.6,
            [0.0, 0.0, 0.0],
            0.0,
        ));
        let yellow_mat = renderer.scene_mut().insert_material(make_material(
            [0.9, 0.85, 0.15, 1.0],
            0.4,
            0.6,
            [0.0, 0.0, 0.0],
            0.0,
        ));
        let white_mat = renderer.scene_mut().insert_material(make_material(
            [0.85, 0.85, 0.9, 1.0],
            0.7,
            0.3,
            [0.0, 0.0, 0.0],
            0.0,
        ));
        let floor_mat = renderer.scene_mut().insert_material(make_material(
            [0.25, 0.25, 0.27, 1.0],
            0.9,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));

        // ── Floor ───────────────────────────────────────────────────────
        let floor = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(plane_mesh(
                [0.0, 0.0, 0.0],
                SCENE_HALF_Z,
            )))
            .as_mesh()
            .unwrap();
        let _ = insert_object(
            &mut renderer,
            floor,
            floor_mat,
            glam::Mat4::IDENTITY,
            SCENE_HALF_Z,
        );

        // ── Tabletop ────────────────────────────────────────────────────
        let table = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(box_mesh(
                [0.0, 0.0, 0.0],
                [8.0, 0.15, SCENE_HALF_Z - 2.0],
            )))
            .as_mesh()
            .unwrap();
        let _ = insert_object(
            &mut renderer,
            table,
            white_mat,
            glam::Mat4::from_translation(glam::Vec3::new(0.0, TABLE_Y, 0.0)),
            SCENE_HALF_Z,
        );

        // ── Objects at varying depths ───────────────────────────────────
        let sphere_m = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(sphere_mesh([0.0; 3], 0.6)))
            .as_mesh()
            .unwrap();
        let box_m = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(cube_mesh([0.0; 3], 0.6)))
            .as_mesh()
            .unwrap();

        let z_positions: [f32; 9] = [-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0];
        let colors = [&red_mat, &green_mat, &blue_mat, &yellow_mat, &white_mat];
        let shapes = [sphere_m, box_m];

        for (i, &z) in z_positions.iter().enumerate() {
            let mat = colors[i % colors.len()];
            let mesh = shapes[i % shapes.len()];
            let x_offset = ((i as f32) - 4.0) * 1.5;
            let _ = insert_object(
                &mut renderer,
                mesh,
                *mat,
                glam::Mat4::from_translation(glam::Vec3::new(x_offset, TABLE_Y + 0.7, z)),
                1.0,
            );
        }

        // ── Light ───────────────────────────────────────────────────────
        let sun = directional_light([0.5, -1.0, -0.3], [1.0, 0.95, 0.9], 3.0);
        renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::light(sun));
        renderer.set_ambient([0.12, 0.12, 0.15], 0.1);

        print_help();

        self.state = Some(AppState {
            window,
            surface,
            device,
            surface_format,
            renderer,
            last_frame: std::time::Instant::now(),
            cam_pos: glam::Vec3::new(0.0, 1.8, 2.0),
            cam_yaw: 0.0,
            cam_pitch: 0.0,
            keys: HashSet::new(),
            just_pressed: Vec::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            scroll_delta: 0.0,
            dof_enabled: true,
            dof_focal_distance: 15.0,
            dof_focal_region: 3.0,
            dof_aperture_blades: 6,
            dof_max_bokeh_size: 30.0,
            dof_aperture_rotation: 0.0,
            dof_sensor_diagonal: 43.3,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
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
                        repeat,
                        ..
                    },
                ..
            } => match ks {
                ElementState::Pressed => {
                    if !repeat {
                        state.just_pressed.push(key);
                    }
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

            WindowEvent::MouseWheel {
                delta: MouseScrollDelta::LineDelta(_, y),
                ..
            } => {
                state.scroll_delta += y;
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
                    color_space: wgpu::SurfaceColorSpace::Auto,
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

fn print_help() {
    println!("\n── Depth of Field Demo ──────────────────────────────────────");
    println!("  WASD/Space/Shift  move      mouse drag  look (click to grab)");
    println!("  Scroll Up/Down    rack focus (focal distance)");
    println!("  Tab               toggle DOF on/off");
    println!("  1/2   focal region +/–   3/4   aperture blades (3-11, 0=Gaussian)");
    println!("  5/6   max bokeh size +/–  7/8   sensor diagonal +/–");
    println!("  9/0   aperture rotation +/–");
    println!("─────────────────────────────────────────────────────────\n");
}

impl AppState {
    fn handle_toggles(&mut self) {
        let pressed = std::mem::take(&mut self.just_pressed);
        let mut changed = false;

        for key in pressed {
            match key {
                KeyCode::Tab => {
                    self.dof_enabled = !self.dof_enabled;
                    changed = true;
                }
                KeyCode::Digit1 => {
                    self.dof_focal_region = (self.dof_focal_region - 1.0).max(0.5);
                    changed = true;
                }
                KeyCode::Digit2 => {
                    self.dof_focal_region = (self.dof_focal_region + 1.0).min(50.0);
                    changed = true;
                }
                KeyCode::Digit3 => {
                    self.dof_aperture_blades = if self.dof_aperture_blades <= 3 {
                        0
                    } else {
                        self.dof_aperture_blades - 1
                    };
                    changed = true;
                }
                KeyCode::Digit4 => {
                    self.dof_aperture_blades = if self.dof_aperture_blades == 0 {
                        3
                    } else {
                        (self.dof_aperture_blades + 1).min(11)
                    };
                    changed = true;
                }
                KeyCode::Digit5 => {
                    self.dof_max_bokeh_size = (self.dof_max_bokeh_size - 5.0).max(1.0);
                    changed = true;
                }
                KeyCode::Digit6 => {
                    self.dof_max_bokeh_size = (self.dof_max_bokeh_size + 5.0).min(100.0);
                    changed = true;
                }
                KeyCode::Digit7 => {
                    self.dof_sensor_diagonal = (self.dof_sensor_diagonal - 5.0).max(10.0);
                    changed = true;
                }
                KeyCode::Digit8 => {
                    self.dof_sensor_diagonal = (self.dof_sensor_diagonal + 5.0).min(100.0);
                    changed = true;
                }
                KeyCode::Digit9 => {
                    self.dof_aperture_rotation -= 0.2;
                    changed = true;
                }
                KeyCode::Digit0 => {
                    self.dof_aperture_rotation += 0.2;
                    changed = true;
                }
                _ => {}
            }
        }

        if changed {
            let mode = if !self.dof_enabled {
                "OFF".to_string()
            } else if self.dof_aperture_blades == 0 {
                "Gaussian".to_string()
            } else {
                format!("Bokeh ({} blades)", self.dof_aperture_blades)
            };
            println!(
                "DOF: {} | focal_dist={:.1} region={:.1} size={:.1} sensor_diag={:.1} rot={:.2}",
                mode,
                self.dof_focal_distance,
                self.dof_focal_region,
                self.dof_max_bokeh_size,
                self.dof_sensor_diagonal,
                self.dof_aperture_rotation,
            );
        }
    }

    fn render(&mut self, dt: f32) {
        const SPEED: f32 = 5.0;
        const LOOK_SENS: f32 = 0.002;

        self.handle_toggles();

        // Scroll-wheel rack focus
        if self.scroll_delta != 0.0 {
            let step = self.scroll_delta * 2.0;
            self.dof_focal_distance = (self.dof_focal_distance + step).clamp(0.5, 50.0);
            self.scroll_delta = 0.0;
        }

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

        let mut camera = Camera::perspective_look_at(
            self.cam_pos,
            self.cam_pos + forward,
            glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            aspect,
            0.1,
            500.0,
        );

        camera.postprocess_settings.dof_enabled = self.dof_enabled;
        camera.postprocess_settings.dof_focal_distance = self.dof_focal_distance;
        camera.postprocess_settings.dof_focal_region = self.dof_focal_region;
        camera.postprocess_settings.dof_aperture_blades = self.dof_aperture_blades;
        camera.postprocess_settings.dof_max_bokeh_size = self.dof_max_bokeh_size;
        camera.postprocess_settings.dof_aperture_rotation = self.dof_aperture_rotation;
        camera.postprocess_settings.dof_sensor_diagonal = self.dof_sensor_diagonal;

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
