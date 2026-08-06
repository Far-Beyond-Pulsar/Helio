//! Sublevels demo — a self-contained structure (a rotating platform with an
//! interior part animating independently) that orbits the main scene at O(1)
//! CPU cost per frame: moving it is two matrix writes
//! (`Scene::move_sublevel` + the derived camera slot Helio updates
//! internally), never a walk over its member objects' transforms.
//!
//! See `docs/portals_and_sublevels.md` for the design and
//! `crates/helio/src/scene/sublevels.rs` for the API this demo drives.
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Mouse drag  — look around (click to grab cursor)
//!   Escape      — release cursor / exit

mod v3_demo_common;

use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera,
    DebugDrawState, GroupId, GroupMask, MaterialId, MeshId, ObjectDescriptor, Renderer,
    RendererConfig, Scene, SublevelDescriptor, SublevelId,
};
use helio_default_graphs::build_default_graph;
use v3_demo_common::{cube_mesh, make_material, plane_mesh, point_light};

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use std::collections::HashSet;
use std::sync::Arc;

/// The sublevel's own group — any id outside the built-in reserved range
/// (`GroupId::EDITOR..=DEBUG`, 0..=7) works.
const SUBLEVEL_GROUP: GroupId = GroupId::new(20);

fn main() {
    env_logger::init();
    log::info!("Starting Helio Sublevels Demo");

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

    // Free-camera state
    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    // Sublevel state
    sublevel: SublevelId,
    sublevel_orbit_center: glam::Vec3,
    /// The interior part's object id — its *local* transform keeps animating
    /// independently of the sublevel's placement, proving the interior
    /// doesn't need to know where the structure is in the world.
    interior_part: helio::ObjectId,
    interior_mesh: MeshId,
    interior_material: MaterialId,
    last_logged_count: u32,
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
                        .with_title("Helio — Sublevels Demo")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .expect("Failed to create window"),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance.create_surface(window.clone()).expect("Failed to create surface");

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

        device.on_uncaptured_error(Arc::new(|e: wgpu::Error| {
            panic!("[GPU UNCAPTURED ERROR] {:?}", e);
        }));
        let info = adapter.get_info();
        println!("[WGPU] Backend: {:?}, Device: {}, Driver: {}", info.backend, info.name, info.driver);
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter().find(|f| f.is_srgb()).copied().unwrap_or(surface_caps.formats[0]);

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

        let mut renderer_config = RendererConfig::new(size.width, size.height, surface_format);
        // The one flag that turns this whole feature on — see
        // docs/portals_and_sublevels.md's "zero overhead when absent"
        // guarantee for why this defaults to `false`.
        renderer_config.enable_sublevels = true;

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
        let graph = build_default_graph(&device, &queue, &scene, renderer_config, debug_state.clone(), &debug_camera_buf, &cull_stats_buf, None);
        let mut renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            renderer_config.surface_format,
            renderer_config.width,
            renderer_config.height,
            renderer_config.render_scale,
            renderer_config,
            scene,
            graph,
            debug_state,
            debug_camera_buf,
            cull_stats_buf,
        );
        renderer.set_editor_mode(true);

        // ── Static world: a floor and a few fixed cubes for scale reference ──
        let floor_mat = renderer.scene_mut().insert_material(make_material([0.35, 0.36, 0.38, 1.0], 0.85, 0.0, [0.0; 3], 0.0));
        let floor_mesh = renderer.scene_mut().insert_mesh(plane_mesh([0.0, 0.0, 0.0], 25.0));
        insert_static_object(&mut renderer, floor_mesh, floor_mat, glam::Mat4::IDENTITY, 25.0);

        let marker_mat = renderer.scene_mut().insert_material(make_material([0.6, 0.15, 0.15, 1.0], 0.6, 0.0, [0.0; 3], 0.0));
        let marker_mesh = renderer.scene_mut().insert_mesh(cube_mesh([0.0, 0.0, 0.0], 0.5));
        for &(x, z) in &[(-8.0, -8.0), (8.0, -8.0), (-8.0, 8.0), (8.0, 8.0)] {
            insert_static_object(&mut renderer, marker_mesh, marker_mat, glam::Mat4::from_translation(glam::Vec3::new(x, 0.5, z)), 0.5);
        }

        // ── The sublevel: a base platform + an interior part, both authored
        // with LOCAL (sublevel-relative) transforms ────────────────────────
        let platform_mat = renderer.scene_mut().insert_material(make_material([0.2, 0.55, 0.85, 1.0], 0.4, 0.6, [0.0; 3], 0.0));
        let platform_mesh = renderer.scene_mut().insert_mesh(cube_mesh([0.0, 0.0, 0.0], 1.5));
        let interior_material = renderer.scene_mut().insert_material(make_material([0.95, 0.75, 0.15, 1.0], 0.25, 0.1, [0.3, 0.2, 0.02], 1.5));
        let interior_mesh = renderer.scene_mut().insert_mesh(cube_mesh([0.0, 0.0, 0.0], 0.4));

        // Base platform sits at the sublevel's local origin.
        insert_sublevel_object(&mut renderer, platform_mesh, platform_mat, glam::Mat4::from_scale(glam::Vec3::new(1.0, 0.2, 1.0)), 2.0);
        // Interior part starts 1 unit above the platform, in local space.
        let interior_part = insert_sublevel_object(
            &mut renderer,
            interior_mesh,
            interior_material,
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 1.0, 0.0)),
            0.6,
        );

        let sublevel_orbit_center = glam::Vec3::new(0.0, 2.0, 0.0);
        let sublevel = renderer.scene_mut().add_sublevel(SublevelDescriptor {
            group: SUBLEVEL_GROUP,
            placement: glam::Mat4::from_translation(sublevel_orbit_center),
            user_tag: 0,
        });
        log::info!(
            "Sublevel created: {} active view slot(s) in use",
            renderer.scene().active_sublevel_count()
        );

        // ── Lighting ─────────────────────────────────────────────────────
        renderer.scene_mut().insert_light(v3_demo_common_sun());
        renderer.scene_mut().insert_light(point_light([0.0, 5.0, 0.0], [1.0, 0.95, 0.85], 8.0, 20.0));

        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_format,
            renderer,
            last_frame: std::time::Instant::now(),
            start_time: std::time::Instant::now(),
            cam_pos: glam::Vec3::new(0.0, 3.5, 12.0),
            cam_yaw: 0.0,
            cam_pitch: -0.15,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            sublevel,
            sublevel_orbit_center,
            interior_part,
            interior_mesh,
            interior_material,
            last_logged_count: 0,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };

        match event {
            WindowEvent::CloseRequested => {
                log::info!("Shutting down");
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::Escape), .. },
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
            WindowEvent::KeyboardInput { event: KeyEvent { state: ks, physical_key: PhysicalKey::Code(key), .. }, .. } => match ks {
                ElementState::Pressed => {
                    state.keys.insert(key);
                }
                ElementState::Released => {
                    state.keys.remove(&key);
                }
            },
            WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => {
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

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: winit::event::DeviceId, event: DeviceEvent) {
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

impl AppState {
    fn render(&mut self, dt: f32) {
        const SPEED: f32 = 6.0;
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
        let time = self.start_time.elapsed().as_secs_f32();

        let camera = Camera::perspective_look_at(self.cam_pos, self.cam_pos + forward, glam::Vec3::Y, std::f32::consts::FRAC_PI_4, aspect, 0.1, 300.0);

        // ── The whole point of sublevels: moving a structure is O(1) ──────
        // Two writes here (`update_sublevel`, the interior part's own
        // `update_object_transform`) regardless of how many objects the
        // sublevel contains — never a walk over its members.
        let orbit_radius = 6.0;
        let orbit_speed = 0.4;
        let orbit_pos = self.sublevel_orbit_center
            + glam::Vec3::new((time * orbit_speed).cos(), (time * 0.6).sin() * 0.5, (time * orbit_speed).sin()) * orbit_radius;
        let placement = glam::Mat4::from_translation(orbit_pos) * glam::Mat4::from_rotation_y(time * 0.5);
        let _ = self.renderer.scene_mut().update_sublevel(self.sublevel, placement);

        // The interior part keeps animating in the sublevel's *local* space
        // — it has no idea the platform is orbiting.
        let bob = (time * 3.0).sin() * 0.4;
        let interior_local = glam::Mat4::from_translation(glam::Vec3::new(0.0, 1.0 + bob, 0.0)) * glam::Mat4::from_rotation_y(time * 2.0);
        let _ = self.renderer.scene_mut().update_object_transform(self.interior_part, interior_local);
        let _ = (self.interior_mesh, self.interior_material); // kept for future re-instancing; silence unused warnings

        let count = self.renderer.scene().active_sublevel_count();
        if count != self.last_logged_count {
            log::info!("Active sublevel view slots: {count}");
            self.last_logged_count = count;
        }

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            _ => return,
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        if let Err(e) = self.renderer.render(&camera, &view) {
            log::error!("Render error: {:?}", e);
        }

        self.queue.present(output);
    }
}

fn v3_demo_common_sun() -> helio::GpuLight {
    v3_demo_common::directional_light([-0.4, -0.8, -0.3], [1.0, 0.97, 0.9], 3.0)
}

fn insert_static_object(renderer: &mut Renderer, mesh: MeshId, material: MaterialId, transform: glam::Mat4, radius: f32) -> helio::ObjectId {
    renderer
        .scene_mut()
        .insert_object(ObjectDescriptor {
            mesh,
            material,
            transform,
            bounds: [transform.w_axis.x, transform.w_axis.y, transform.w_axis.z, radius],
            flags: 0,
            groups: GroupMask::NONE,
            movability: None,
            user_tag: 0,
        })
        .expect("insert static object")
}

/// Inserts an object into the sublevel's group with a **local** transform —
/// `transform` is relative to the sublevel's own origin, not world space
/// (see `SublevelDescriptor::group`'s doc comment). Must be `Movable` so the
/// interior part can keep animating.
fn insert_sublevel_object(renderer: &mut Renderer, mesh: MeshId, material: MaterialId, local_transform: glam::Mat4, radius: f32) -> helio::ObjectId {
    renderer
        .scene_mut()
        .insert_object(ObjectDescriptor {
            mesh,
            material,
            transform: local_transform,
            bounds: [local_transform.w_axis.x, local_transform.w_axis.y, local_transform.w_axis.z, radius],
            flags: 0,
            groups: GroupMask::from(SUBLEVEL_GROUP),
            movability: Some(helio::Movability::Movable),
            user_tag: 0,
        })
        .expect("insert sublevel object")
}
