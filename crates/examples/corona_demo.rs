//! Corona — GPU particle system demo.
//!
//! Showcases 4 distinct emitter configurations at scale (1M total particles):
//!   1. Fountain  (Point, upward velocity, gravity, blue→cyan fade)
//!   2. Nebula    (Sphere, slow orbit, purple→pink)
//!   3. FireRing  (Sphere, radial outward, orange→red)
//!   4. Galaxy    (Sphere, spinning disk, white→blue)
//!
//! Controls:
//!   WASD        — fly
//!   Space/Shift — up/down
//!   Mouse       — look (click to grab)
//!   Escape      — release / quit

mod v3_demo_common;

use std::collections::HashSet;
use std::sync::Arc;

use glam::{EulerRot, Mat4, Quat, Vec3};
use helio::{
    required_wgpu_features, required_wgpu_limits, Camera, DebugDrawState, GpuLight, LightType, Renderer,
    RendererConfig, Scene, SceneActor, SkyActor,
};
use helio_default_graphs::build_default_graph;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

// ── constants ─────────────────────────────────────────────────────────────────

const LOOK_SENS: f32 = 0.002;
const FLY_SPEED: f32 = 5.0;
const DRAG: f32 = 8.0;

// ── app ───────────────────────────────────────────────────────────────────────

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
    // fly camera
    cam_pos: Vec3,
    yaw: f32,
    pitch: f32,
    velocity: Vec3,
    // input
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),
    // emitter descriptors (CPU-side, rebuilt each frame with position/rotation)
    emitters: [libhelio::GpuCoronaEmitter; 4],
}

impl App {
    fn new() -> Self {
        Self { state: None }
    }

    fn build_emitters(elapsed: f32) -> [libhelio::GpuCoronaEmitter; 4] {
        let t = elapsed;

        // ── 1. Fountain: blue fountain bursting from origin ────────────────
        let fountain = libhelio::CoronaEmitterDescriptor {
            max_particles: 262_144,
            emit_rate: 8000.0,
            lifetime: 3.0,
            lifetime_variation: 1.0,
            start_size: [0.8, 0.8],
            end_size: [0.05, 0.05],
            start_color: [0.2, 0.6, 1.0, 1.0],
            end_color: [0.0, 1.0, 1.0, 0.0],
            velocity: [0.0, 8.0, 0.0],
            velocity_variation: [3.0, 2.0, 3.0],
            gravity: -5.0,
            shape: libhelio::CoronaEmitterShape::Point,
            texture_index: -1,
            position: [0.0, 0.0, 0.0],
        };

        // ── 2. Nebula: slow purple cloud orbiting origin ───────────────────
        let nebula_pos = [
            6.0 * t.cos() * 0.5,
            2.0 + 1.5 * (t * 0.7).sin(),
            6.0 * t.sin() * 0.5,
        ];
        let nebula = libhelio::CoronaEmitterDescriptor {
            max_particles: 131_072,
            emit_rate: 3000.0,
            lifetime: 6.0,
            lifetime_variation: 2.0,
            start_size: [1.5, 1.5],
            end_size: [3.0, 3.0],
            start_color: [0.8, 0.2, 0.8, 0.6],
            end_color: [0.3, 0.1, 0.6, 0.0],
            velocity: [0.0, 0.0, 0.0],
            velocity_variation: [0.5, 0.5, 0.5],
            gravity: 0.0,
            shape: libhelio::CoronaEmitterShape::Sphere { radius: 2.5 },
            texture_index: -1,
            position: nebula_pos,
        };

        // ── 3. FireRing: radial burst in XZ plane ──────────────────────────
        let ring_angle = t * 0.6;
        let ring_pos = [
            4.0 * ring_angle.cos(),
            0.5,
            4.0 * ring_angle.sin(),
        ];
        let fire = libhelio::CoronaEmitterDescriptor {
            max_particles: 65_536,
            emit_rate: 2000.0,
            lifetime: 1.5,
            lifetime_variation: 0.5,
            start_size: [0.6, 0.6],
            end_size: [0.0, 0.0],
            start_color: [1.0, 0.6, 0.0, 1.0],
            end_color: [1.0, 0.0, 0.0, 0.0],
            velocity: [
                3.0 * ring_angle.cos(),
                0.0,
                3.0 * ring_angle.sin(),
            ],
            velocity_variation: [2.0, 1.0, 2.0],
            gravity: -2.0,
            shape: libhelio::CoronaEmitterShape::Sphere { radius: 0.8 },
            texture_index: -1,
            position: ring_pos,
        };

        // ── 4. Galaxy: spinning disk of white-blue particles ───────────────
        let galaxy = libhelio::CoronaEmitterDescriptor {
            max_particles: 131_072,
            emit_rate: 4000.0,
            lifetime: 8.0,
            lifetime_variation: 3.0,
            start_size: [0.4, 0.4],
            end_size: [0.8, 0.8],
            start_color: [0.7, 0.8, 1.0, 0.8],
            end_color: [0.2, 0.3, 0.6, 0.0],
            velocity: [0.0, 0.0, 0.0],
            velocity_variation: [0.0, 0.0, 0.0],
            gravity: 0.0,
            shape: libhelio::CoronaEmitterShape::Sphere { radius: 6.0 },
            texture_index: -1,
            position: [0.0, 8.0, 0.0],
        };

        [
            fountain.to_gpu(),
            nebula.to_gpu(),
            fire.to_gpu(),
            galaxy.to_gpu(),
        ]
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
                        .with_title("Helio — Corona GPU Particle System Demo")
                        .with_inner_size(winit::dpi::LogicalSize::new(1600u32, 900u32)),
                )
                .expect("create window"),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: true,
        }))
        .expect("no adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                required_features: required_wgpu_features(adapter.features()),
                required_limits: required_wgpu_limits(adapter.limits()),
                ..Default::default()
            }))
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
                color_space: wgpu::SurfaceColorSpace::Auto,
            },
        );

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
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let debug_state = Arc::new(std::sync::Mutex::new(DebugDrawState::default()));
        let graph = build_default_graph(&device, &queue, &scene, config, debug_state.clone(), &debug_camera_buf, &cull_stats_buf, None);
        let mut renderer = Renderer::new(
            device.clone(), queue.clone(),
            config.surface_format, config.width, config.height, config.render_scale,
            config, scene, graph, debug_state, debug_camera_buf, cull_stats_buf,
        );

        // ── Sky + lighting ───────────────────────────────────────────────────
        renderer.scene_mut().insert_actor(SceneActor::sky(
            SkyActor::new().with_sky_color([0.08, 0.10, 0.20]),
        ));
        renderer.scene_mut().insert_actor(SceneActor::light(GpuLight {
            position_range: [0.0, 0.0, 0.0, f32::MAX],
            direction_outer: [-0.3, -1.0, -0.5, 0.0],
            color_intensity: [0.9, 0.85, 0.75, 3.0],
            shadow_index: 0,
            light_type: LightType::Directional as u32,
            inner_angle: 0.0,
            _pad: 0,
        }));
        renderer.set_ambient([0.08, 0.10, 0.18], 0.6);
        renderer.set_clear_color([0.02, 0.03, 0.08, 1.0]);

        // ── Floor plane ─────────────────────────────────────────────────────
        let floor_mesh_id = renderer.scene_mut().insert_actor(SceneActor::mesh(
            v3_demo_common::plane_mesh([0.0, 0.0, 0.0], 30.0),
        )).as_mesh().unwrap();
        let floor_mat = renderer.scene_mut().insert_material(v3_demo_common::make_material(
            [0.06, 0.06, 0.08, 1.0],
            0.8,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));
        renderer.scene_mut().insert_actor(SceneActor::object(helio::ObjectDescriptor {
            mesh: floor_mesh_id,
            material: floor_mat,
            transform: Mat4::from_translation(glam::Vec3::new(0.0, -0.5, 0.0)),
            bounds: [0.0, -0.5, 0.0, 43.0],
            flags: 0,
            groups: helio::GroupMask::NONE,
            movability: None,
            user_tag: 0,
        }));

        // Build initial emitters
        let emitters = Self::build_emitters(0.0);
        renderer.set_corona_emitters(&emitters);

        self.state = Some(AppState {
            window,
            surface,
            device,
            surface_format,
            renderer,
            last_frame: std::time::Instant::now(),
            start_time: std::time::Instant::now(),
            cam_pos: Vec3::new(0.0, 4.0, 12.0),
            yaw: 0.0,
            pitch: -0.3,
            velocity: Vec3::ZERO,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            emitters,
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

                // Update fly camera
                let (dx, dy) = state.mouse_delta;
                state.mouse_delta = (0.0, 0.0);
                state.yaw -= dx * LOOK_SENS;
                state.pitch = (state.pitch - dy * LOOK_SENS).clamp(-1.5, 1.5);
                let orientation = Quat::from_euler(EulerRot::YXZ, state.yaw, state.pitch, 0.0);
                let forward = orientation * -Vec3::Z;
                let right = orientation * Vec3::X;
                let up = Vec3::Y;
                let mut accel = Vec3::ZERO;
                if state.keys.contains(&KeyCode::KeyW) { accel += forward; }
                if state.keys.contains(&KeyCode::KeyS) { accel -= forward; }
                if state.keys.contains(&KeyCode::KeyA) { accel -= right; }
                if state.keys.contains(&KeyCode::KeyD) { accel += right; }
                if state.keys.contains(&KeyCode::Space) { accel += up; }
                if state.keys.contains(&KeyCode::ShiftLeft) { accel -= up; }
                if accel.length_squared() > 0.0 { accel = accel.normalize(); }
                state.velocity += accel * FLY_SPEED * dt;
                state.velocity /= 1.0 + DRAG * dt;
                state.cam_pos += state.velocity * dt;

                let size = state.window.inner_size();
                let camera = Camera::perspective_look_at(
                    state.cam_pos,
                    state.cam_pos + orientation * -Vec3::Z,
                    orientation * Vec3::Y,
                    std::f32::consts::FRAC_PI_4,
                    size.width as f32 / size.height.max(1) as f32,
                    0.01,
                    200.0,
                );

                // ── Update emitters with time-varying positions ──────────────
                let elapsed = state.start_time.elapsed().as_secs_f32();
                let new_emitters = App::build_emitters(elapsed);
                state.renderer.set_corona_emitters(&new_emitters);

                // ── Render ────────────────────────────────────────────────────
                let output = match state.surface.get_current_texture() {
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
                if let Err(e) = state.renderer.render(&camera, &view) {
                    log::error!("render error: {:?}", e);
                }
                state.renderer.queue().present(output);
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
    log::info!("=== Corona GPU Particle System Demo ===");
    log::info!("590,824 total particles across 4 emitters");
    log::info!("Fly around to see the full effect!");
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("event loop error");
}
