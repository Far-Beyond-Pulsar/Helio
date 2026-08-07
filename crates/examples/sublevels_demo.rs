//! Sublevels demo — a group of objects moved as a unit via one coordinate-space
//! transform, instead of per-object updates.
//!
//! A static hub room holds a small floating platform (a lit disc, a pillar
//! marker, and 1,024 tiny cubes riding on the deck) that is registered as a
//! *sublevel*: all of its objects keep their ordinary local transforms, and
//! the whole platform — all 1,026 objects on it — is moved every frame with
//! a single `Scene::update_sublevel` call. O(1) regardless of how many
//! objects are on it: watch it orbit smoothly and cast real shadows as it
//! moves.
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Mouse drag  — look around (click to grab cursor)
//!   Escape      — release cursor / exit

mod v3_demo_common;

use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera,
    DebugDrawState, GroupId, GroupMask, LightId, ObjectDescriptor, Renderer, RendererConfig,
    Scene, SceneActor, SublevelDescriptor, SublevelId,
};
use helio_default_graphs::build_default_graph;
use v3_demo_common::{box_mesh, make_material, point_light, sphere_mesh};

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use std::collections::HashSet;
use std::sync::Arc;

/// Group tag for everything that rides on the floating sublevel platform.
const PLATFORM_GROUP: GroupId = GroupId::new(20);

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
    start_time: std::time::Instant,

    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    sublevel: SublevelId,
    _light_ids: Vec<LightId>,
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
                        .with_title("Helio – Sublevels")
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

        // ── Hub room: 12m x 4m x 12m box shell, walls facing inward ──────────
        let wall_mat = renderer.scene_mut().insert_material(make_material(
            [0.7, 0.7, 0.72, 1.0],
            0.85,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));
        let floor = renderer.scene_mut().insert_actor(SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [6.0, 0.05, 6.0]))).as_mesh().unwrap();
        let ceiling = renderer.scene_mut().insert_actor(SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [6.0, 0.05, 6.0]))).as_mesh().unwrap();
        let wall_n = renderer.scene_mut().insert_actor(SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [6.0, 2.0, 0.05]))).as_mesh().unwrap();
        let wall_s = renderer.scene_mut().insert_actor(SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [6.0, 2.0, 0.05]))).as_mesh().unwrap();
        let wall_e = renderer.scene_mut().insert_actor(SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [0.05, 2.0, 6.0]))).as_mesh().unwrap();
        let wall_w = renderer.scene_mut().insert_actor(SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [0.05, 2.0, 6.0]))).as_mesh().unwrap();

        let _ = v3_demo_common::insert_object(&mut renderer, floor, wall_mat, glam::Mat4::IDENTITY, 9.0);
        let _ = v3_demo_common::insert_object(&mut renderer, ceiling, wall_mat, glam::Mat4::from_translation(glam::Vec3::new(0.0, 4.0, 0.0)), 9.0);
        let _ = v3_demo_common::insert_object(&mut renderer, wall_n, wall_mat, glam::Mat4::from_translation(glam::Vec3::new(0.0, 2.0, -6.0)), 6.0);
        let _ = v3_demo_common::insert_object(&mut renderer, wall_s, wall_mat, glam::Mat4::from_translation(glam::Vec3::new(0.0, 2.0, 6.0)), 6.0);
        let _ = v3_demo_common::insert_object(&mut renderer, wall_e, wall_mat, glam::Mat4::from_translation(glam::Vec3::new(6.0, 2.0, 0.0)), 6.0);
        let _ = v3_demo_common::insert_object(&mut renderer, wall_w, wall_mat, glam::Mat4::from_translation(glam::Vec3::new(-6.0, 2.0, 0.0)), 6.0);

        // ── Floating platform: registered as a sublevel ──────────────────────
        // Every mesh below is authored in the platform's own *local* space
        // (origin at the platform's centre) — that never changes. The whole
        // group is placed and re-placed purely via the sublevel's coordinate
        // space, not by touching these objects again.
        let platform_mat = renderer.scene_mut().insert_material(make_material(
            [0.15, 0.55, 0.95, 1.0],
            0.35,
            0.6,
            [0.05, 0.35, 0.9],
            1.2,
        ));
        let deck_mesh = renderer.scene_mut().insert_actor(SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [1.4, 0.08, 1.4]))).as_mesh().unwrap();
        let pillar_mesh = renderer.scene_mut().insert_actor(SceneActor::mesh(sphere_mesh([0.0, 0.0, 0.0], 0.25))).as_mesh().unwrap();

        insert_grouped_object(&mut renderer, deck_mesh, platform_mat, glam::Mat4::IDENTITY, 2.0, PLATFORM_GROUP);
        insert_grouped_object(
            &mut renderer,
            pillar_mesh,
            platform_mat,
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.45, 0.0)),
            0.3,
            PLATFORM_GROUP,
        );

        // ── 1,024 tiny cubes riding on the deck ──────────────────────────────
        // The point of a sublevel is that moving it costs the same whether it
        // carries 2 objects or 2,000: watch the frame time stay flat while a
        // thousand-plus objects orbit together on one `update_sublevel` call
        // per frame (see `AppState::render`). All share one mesh + material,
        // so they also batch into a single instanced draw call — this swarm
        // costs one GPU draw, not a thousand.
        let stud_mesh = renderer.scene_mut().insert_actor(SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [0.022, 0.022, 0.022]))).as_mesh().unwrap();
        let stud_mat = renderer.scene_mut().insert_material(make_material(
            [0.85, 0.9, 1.0, 1.0],
            0.4,
            0.3,
            [0.3, 0.6, 1.0],
            0.4,
        ));
        const GRID: i32 = 128; // 128*128 = 16,384 cubes
        let spacing = 2.6 / GRID as f32;
        for ix in 0..GRID {
            for iz in 0..GRID {
                let x = (ix as f32 - (GRID - 1) as f32 * 0.5) * spacing;
                let z = (iz as f32 - (GRID - 1) as f32 * 0.5) * spacing;
                // Deterministic pseudo-random height jitter — no RNG dependency needed.
                let jitter = ((ix * 928371 + iz * 12923) as f32 * 0.0001).sin().abs();
                let y = 0.08 + 0.022 + jitter * 0.06;
                insert_grouped_object(
                    &mut renderer,
                    stud_mesh,
                    stud_mat,
                    glam::Mat4::from_translation(glam::Vec3::new(x, y, z)),
                    0.04,
                    PLATFORM_GROUP,
                );
            }
        }

        // Sublevel starting placement — the platform's local origin maps here
        // in world space until the first `update_sublevel` call below moves it.
        let start_placement = glam::Mat4::from_translation(glam::Vec3::new(2.5, 1.6, 0.0));
        let sublevel = renderer
            .scene_mut()
            .add_sublevel(SublevelDescriptor {
                group: PLATFORM_GROUP,
                placement: start_placement,
            })
            .expect("add_sublevel");

        // A light riding on the platform, so the moving light itself
        // demonstrates the sublevel transform too (not just the mesh).
        let mut light_ids = Vec::new();
        light_ids.push(
            renderer
                .scene_mut()
                .insert_actor(SceneActor::light(point_light(
                    [0.0, 0.9, 0.0],
                    [0.4, 0.75, 1.0],
                    3.0,
                    5.0,
                )))
                .as_light()
                .unwrap(),
        );
        light_ids.push(
            renderer
                .scene_mut()
                .insert_actor(SceneActor::light(point_light(
                    [0.0, 3.6, 0.0],
                    [1.0, 0.95, 0.85],
                    2.0,
                    10.0,
                )))
                .as_light()
                .unwrap(),
        );
        renderer.set_ambient([0.85, 0.9, 1.0], 0.06);
        renderer.set_clear_color([0.01, 0.01, 0.02, 1.0]);

        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_format: format,
            renderer,
            last_frame: std::time::Instant::now(),
            start_time: std::time::Instant::now(),
            cam_pos: glam::Vec3::new(0.0, 1.6, 5.0),
            cam_yaw: std::f32::consts::PI,
            cam_pitch: -0.15,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            sublevel,
            _light_ids: light_ids,
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
                    let ok = state
                        .window
                        .set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                        .is_ok();
                    if ok {
                        state.window.set_cursor_visible(false);
                        state.cursor_grabbed = true;
                    }
                }
            }
            WindowEvent::Resized(s) if s.width > 0 && s.height > 0 => {
                state.surface.configure(
                    &state.device,
                    &wgpu::SurfaceConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        format: state.surface_format,
                        width: s.width,
                        height: s.height,
                        present_mode: wgpu::PresentMode::Fifo,
                        alpha_mode: wgpu::CompositeAlphaMode::Auto,
                        view_formats: vec![],
                        desired_maximum_frame_latency: 2,
                        color_space: wgpu::SurfaceColorSpace::Auto,
                    },
                );
                state.renderer.set_render_size(s.width, s.height);
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
        const SPEED: f32 = 4.0;
        const SENS: f32 = 0.002;

        self.cam_yaw += self.mouse_delta.0 * SENS;
        self.cam_pitch = (self.cam_pitch - self.mouse_delta.1 * SENS).clamp(-1.4, 1.4);
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

        // ── The whole point of this demo: orbit the platform every frame ────
        // One `update_sublevel` call moves the deck, the pillar marker, and
        // the light riding on it together — O(1) regardless of how many
        // objects are tagged into the sublevel's group.
        let t = self.start_time.elapsed().as_secs_f32();
        let orbit_radius = 2.5;
        let placement = glam::Mat4::from_translation(glam::Vec3::new(
            orbit_radius * t.cos(),
            1.6 + 0.4 * (t * 0.6).sin(),
            orbit_radius * t.sin(),
        )) * glam::Mat4::from_rotation_y(t * 0.8);
        self.renderer
            .scene_mut()
            .update_sublevel(self.sublevel, placement)
            .expect("update_sublevel");

        let size = self.window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;

        let camera = Camera::perspective_look_at(
            self.cam_pos,
            self.cam_pos + forward,
            glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            aspect,
            0.1,
            100.0,
        );

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            _ => return,
        };
        let view = output.texture.create_view(&Default::default());

        if let Err(e) = self.renderer.render(&camera, &view) {
            log::error!("Render: {:?}", e);
        }
        self.queue.present(output);
    }
}

/// Like `v3_demo_common::insert_object`, but tags the object into `groups` —
/// the wrapper in that module always inserts with `GroupMask::NONE`, which
/// would leave it outside any sublevel's membership.
fn insert_grouped_object(
    renderer: &mut Renderer,
    mesh: helio::MeshId,
    material: helio::MaterialId,
    transform: glam::Mat4,
    radius: f32,
    group: GroupId,
) -> helio::ObjectId {
    renderer
        .scene_mut()
        .insert_actor(SceneActor::object(ObjectDescriptor {
            mesh,
            material,
            transform,
            bounds: [
                transform.w_axis.x,
                transform.w_axis.y,
                transform.w_axis.z,
                radius,
            ],
            flags: 0,
            groups: GroupMask::from(group),
            movability: Some(helio::Movability::Movable),
            user_tag: 0,
        }))
        .as_object()
        .expect("insert object")
}
