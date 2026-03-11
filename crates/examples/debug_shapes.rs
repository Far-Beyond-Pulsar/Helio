//! Debug shape visualization example
//!
//! Demonstrates all debug drawing primitives:
//! - Lines (rendered as 3D tubes)
//! - Spheres (3 orthogonal circles)
//! - Boxes (12 wireframe edges)
//! - Cones (base circle + radial spokes)
//! - Capsules (2 hemisphere caps + 4 edges)
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Mouse drag  — look around (click to grab cursor)
//!   Escape      — release cursor / exit



mod demo_portal;

use helio_render_v2::{Renderer, RendererConfig, Camera};


use helio_render_v2::features::FeatureRegistry;


use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId, CursorGrabMode},
};


use std::collections::HashSet;


use std::sync::Arc;

fn main() {
    env_logger::init();
    log::info!("Starting Helio Debug Shapes Example");

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

    // Free-camera state
    cam_pos:   glam::Vec3,
    cam_yaw:   f32,
    cam_pitch: f32,
    keys:      HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    // Animation
    time: f32,
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
                        .with_title("Helio Debug Shapes Example")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .expect("Failed to create window"),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::VALIDATION | wgpu::InstanceFlags::GPU_BASED_VALIDATION | wgpu::InstanceFlags::DEBUG,
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

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Main Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            },
        ))
        .expect("Failed to create device");

        device.on_uncaptured_error(std::sync::Arc::new(|e| {
            panic!("[GPU UNCAPTURED ERROR] {:?}", e);
        }));
        let info = adapter.get_info();
        println!("[WGPU] Backend: {:?}, Device: {}, Driver: {}", info.backend, info.name, info.driver);
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

        // Minimal feature registry - we only need debug drawing
        let feature_registry = FeatureRegistry::builder().build();

        let mut renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            RendererConfig::new(size.width, size.height, surface_format, feature_registry),
        )
        .expect("Failed to create renderer");
        renderer.set_editor_mode(true);
        demo_portal::enable_live_dashboard(&mut renderer);

        self.state = Some(AppState {
            window,
            surface,
            device,
            surface_format,
            renderer,
            last_frame: std::time::Instant::now(),
            cam_pos:   glam::Vec3::new(0.0, 3.0, 10.0),
            cam_yaw:   0.0,
            cam_pitch: -0.15,
            keys:      HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            time: 0.0,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    let config = wgpu::SurfaceConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        format: state.surface_format,
                        width: new_size.width,
                        height: new_size.height,
                        present_mode: wgpu::PresentMode::Fifo,
                        alpha_mode: wgpu::CompositeAlphaMode::Auto,
                        view_formats: vec![],
                        desired_maximum_frame_latency: 2,
                    };
                    state.surface.configure(&state.device, &config);
                    state.renderer.resize(new_size.width, new_size.height);
                }
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(code),
                    state: key_state,
                    ..
                },
                ..
            } => {
                match key_state {
                    ElementState::Pressed => {
                        if code == KeyCode::F3 {
                        state.renderer.debug_viz_mut().enabled ^= true;
                    }
                    state.keys.insert(code);
                        if code == KeyCode::Escape {
                            if state.cursor_grabbed {
                                let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                                state.window.set_cursor_visible(true);
                                state.cursor_grabbed = false;
                            } else {
                                event_loop.exit();
                            }
                        }
                    }
                    ElementState::Released => {
                        state.keys.remove(&code);
                    }
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if !state.cursor_grabbed {
                    if state
                        .window
                        .set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                        .is_ok()
                    {
                        state.window.set_cursor_visible(false);
                        state.cursor_grabbed = true;
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt = now.duration_since(state.last_frame).as_secs_f32();
                state.last_frame = now;
                state.time += dt;

                state.update_camera(dt);
                state.draw_debug_shapes();
                state.render();

                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _device_id: DeviceId, event: DeviceEvent) {
        let Some(state) = &mut self.state else { return };
        if let DeviceEvent::MouseMotion { delta } = event {
            if state.cursor_grabbed {
                state.mouse_delta.0 += delta.0 as f32;
                state.mouse_delta.1 += delta.1 as f32;
            }
        }
    }
}

impl AppState {
    fn update_camera(&mut self, dt: f32) {
        const LOOK_SPEED: f32 = 0.003;
        const MOVE_SPEED: f32 = 5.0;

        // Mouse look
        if self.cursor_grabbed {
            self.cam_yaw   += self.mouse_delta.0 * LOOK_SPEED;
            self.cam_pitch -= self.mouse_delta.1 * LOOK_SPEED;
            self.cam_pitch = self.cam_pitch.clamp(-std::f32::consts::FRAC_PI_2 * 0.99, std::f32::consts::FRAC_PI_2 * 0.99);
            self.mouse_delta = (0.0, 0.0);
        }

        // Movement
        let fwd = glam::Vec3::new(
            self.cam_yaw.sin(),
            0.0,
            -self.cam_yaw.cos(),
        ).normalize();
        let right = glam::Vec3::new(self.cam_yaw.cos(), 0.0, self.cam_yaw.sin());
        let up = glam::Vec3::Y;

        let mut vel = glam::Vec3::ZERO;
        if self.keys.contains(&KeyCode::KeyW) { vel += fwd; }
        if self.keys.contains(&KeyCode::KeyS) { vel -= fwd; }
        if self.keys.contains(&KeyCode::KeyD) { vel += right; }
        if self.keys.contains(&KeyCode::KeyA) { vel -= right; }
        if self.keys.contains(&KeyCode::Space) { vel += up; }
        if self.keys.contains(&KeyCode::ShiftLeft) { vel -= up; }

        if vel.length_squared() > 0.0 {
            self.cam_pos += vel.normalize() * MOVE_SPEED * dt;
        }
    }

    fn draw_debug_shapes(&mut self) {
        use std::f32::consts::PI;
        use glam::Vec3;

        let t = self.time;

        // === Lines: Create a grid and some animated lines ===
        
        // Grid on the ground plane
        for i in -5..=5 {
            let x = i as f32;
            let color = if i == 0 { [0.6, 0.6, 0.6, 1.0] } else { [0.3, 0.3, 0.3, 1.0] };
            self.renderer.debug_line(Vec3::new(x, 0.0, -5.0), Vec3::new(x, 0.0, 5.0), color, 0.01);
            self.renderer.debug_line(Vec3::new(-5.0, 0.0, x), Vec3::new(5.0, 0.0, x), color, 0.01);
        }

        // Spinning line above origin
        let angle = t * 0.8;
        let r = 2.0;
        let p1 = Vec3::new(r * angle.cos(), 2.0, r * angle.sin());
        let p2 = Vec3::new(-r * angle.cos(), 2.0, -r * angle.sin());
        self.renderer.debug_line(p1, p2, [1.0, 0.2, 0.2, 1.0], 0.05);

        // === Spheres: Show different colors and sizes ===
        
        // Static sphere at origin (reference point)
        self.renderer.debug_sphere(Vec3::new(0.0, 0.5, 0.0), 0.5, [0.2, 1.0, 0.2, 0.8], 0.02);

        // Orbiting spheres
        for i in 0..3 {
            let orbit_angle = t * 0.5 + i as f32 * 2.0 * PI / 3.0;
            let orbit_radius = 3.0;
            let height = 1.5 + (t * 0.7 + i as f32).sin() * 0.5;
            let pos = Vec3::new(
                orbit_radius * orbit_angle.cos(),
                height,
                orbit_radius * orbit_angle.sin(),
            );
            let colors = [
                [1.0, 0.5, 0.0, 0.9],  // Orange
                [0.0, 0.7, 1.0, 0.9],  // Cyan
                [1.0, 0.0, 1.0, 0.9],  // Magenta
            ];
            self.renderer.debug_sphere(pos, 0.3, colors[i], 0.02);
        }

        // === Boxes: Show rotation and different orientations ===
        
        // Rotating box (center, half_extents, rotation)
        let box_rot = glam::Quat::from_euler(glam::EulerRot::XYZ, t * 0.3, t * 0.5, t * 0.7);
        self.renderer.debug_box(
            Vec3::new(-3.0, 1.5, 0.0),
            Vec3::new(0.4, 0.4, 0.4),  // half extents
            box_rot,
            [1.0, 1.0, 0.0, 1.0],
            0.03,
        );

        // Static boxes showing different scales
        self.renderer.debug_box(
            Vec3::new(3.0, 0.6, 2.0),
            Vec3::new(0.3, 0.6, 0.2),  // half extents
            glam::Quat::IDENTITY,
            [0.0, 1.0, 1.0, 1.0],
            0.02,
        );

        // === Cones: Show direction indicators (apex, direction, height, radius) ===
        
        // Spinning cones pointing upward
        for i in 0..4 {
            let cone_angle = t * 0.4 + i as f32 * PI * 0.5;
            let cone_radius = 4.0;
            let apex = Vec3::new(
                cone_radius * cone_angle.cos(),
                0.5,
                cone_radius * cone_angle.sin(),
            );
            let direction = Vec3::Y;  // pointing up
            let height = 1.5;
            let base_radius = 0.3;
            
            let colors = [
                [1.0, 0.0, 0.0, 0.8],  // Red
                [0.0, 1.0, 0.0, 0.8],  // Green
                [0.0, 0.0, 1.0, 0.8],  // Blue
                [1.0, 1.0, 0.0, 0.8],  // Yellow
            ];
            
            self.renderer.debug_cone(apex, direction, height, base_radius, colors[i], 0.02);
        }

        // === Capsules: Show orientation and scale ===
        
        // Tumbling capsule
        let capsule_rot = glam::Quat::from_euler(glam::EulerRot::XYZ, t * 0.6, 0.0, t * 0.4);
        let offset = Vec3::new(0.0, 0.0, 1.0);
        let cap_start = Vec3::new(0.0, 3.5, -3.0) + capsule_rot * offset;
        let cap_end = Vec3::new(0.0, 3.5, -3.0) + capsule_rot * -offset;
        self.renderer.debug_capsule(cap_start, cap_end, 0.4, [1.0, 0.5, 1.0, 0.9], 0.025);

        // Horizontal capsule
        self.renderer.debug_capsule(
            Vec3::new(2.0, 0.5, -3.0),
            Vec3::new(4.0, 0.5, -3.0),
            0.25,
            [0.5, 1.0, 0.5, 0.9],
            0.02,
        );

        // Vertical capsule
        self.renderer.debug_capsule(
            Vec3::new(-4.0, 0.0, -2.0),
            Vec3::new(-4.0, 2.0, -2.0),
            0.3,
            [0.5, 0.5, 1.0, 0.9],
            0.02,
        );

        // === Complex composed shapes ===
        
        // Coordinate frame at origin
        let axis_len = 1.0;
        self.renderer.debug_line(Vec3::ZERO, Vec3::new(axis_len, 0.0, 0.0), [1.0, 0.0, 0.0, 1.0], 0.04);  // X = Red
        self.renderer.debug_line(Vec3::ZERO, Vec3::new(0.0, axis_len, 0.0), [0.0, 1.0, 0.0, 1.0], 0.04);  // Y = Green
        self.renderer.debug_line(Vec3::ZERO, Vec3::new(0.0, 0.0, axis_len), [0.0, 0.0, 1.0, 1.0], 0.04);  // Z = Blue

        // Bounding box visualization (simulating physics debug)
        let bbox_min = Vec3::new(-2.5, 0.0, 2.0);
        let bbox_max = Vec3::new(-1.5, 1.0, 3.0);
        draw_aabb(&mut self.renderer, bbox_min, bbox_max, [1.0, 0.5, 0.0, 0.7], 0.015);
    }

    fn render(&mut self) {
        let output = match self.surface.get_current_texture() {
            Ok(texture) => texture,
            Err(e) => {
                log::warn!("Failed to acquire next swap chain texture: {:?}", e);
                return;
            }
        };

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Build camera from free-fly state
        let fwd = glam::Vec3::new(
            self.cam_yaw.sin() * self.cam_pitch.cos(),
            self.cam_pitch.sin(),
            -self.cam_yaw.cos() * self.cam_pitch.cos(),
        ).normalize();
        let size = self.window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;
        let camera = Camera::perspective(
            self.cam_pos,
            self.cam_pos + fwd,
            glam::Vec3::Y,
            70.0_f32.to_radians(),
            aspect,
            0.1,
            1000.0,
            self.time,
        );

        // Scene state is persistent — no per-frame setup needed.
        self.renderer.render(&camera, &view, 0.016).ok();

        output.present();
    }
}

/// Helper function to draw an axis-aligned bounding box using debug lines
fn draw_aabb(renderer: &mut Renderer, min: glam::Vec3, max: glam::Vec3, color: [f32; 4], thickness: f32) {
    let x0 = min.x;
    let y0 = min.y;
    let z0 = min.z;
    let x1 = max.x;
    let y1 = max.y;
    let z1 = max.z;

    use glam::Vec3;

    // Bottom face (y = y0)
    renderer.debug_line(Vec3::new(x0, y0, z0), Vec3::new(x1, y0, z0), color, thickness);
    renderer.debug_line(Vec3::new(x1, y0, z0), Vec3::new(x1, y0, z1), color, thickness);
    renderer.debug_line(Vec3::new(x1, y0, z1), Vec3::new(x0, y0, z1), color, thickness);
    renderer.debug_line(Vec3::new(x0, y0, z1), Vec3::new(x0, y0, z0), color, thickness);

    // Top face (y = y1)
    renderer.debug_line(Vec3::new(x0, y1, z0), Vec3::new(x1, y1, z0), color, thickness);
    renderer.debug_line(Vec3::new(x1, y1, z0), Vec3::new(x1, y1, z1), color, thickness);
    renderer.debug_line(Vec3::new(x1, y1, z1), Vec3::new(x0, y1, z1), color, thickness);
    renderer.debug_line(Vec3::new(x0, y1, z1), Vec3::new(x0, y1, z0), color, thickness);

    // Vertical edges
    renderer.debug_line(Vec3::new(x0, y0, z0), Vec3::new(x0, y1, z0), color, thickness);
    renderer.debug_line(Vec3::new(x1, y0, z0), Vec3::new(x1, y1, z0), color, thickness);
    renderer.debug_line(Vec3::new(x1, y0, z1), Vec3::new(x1, y1, z1), color, thickness);
    renderer.debug_line(Vec3::new(x0, y0, z1), Vec3::new(x0, y1, z1), color, thickness);
}
