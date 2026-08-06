//! Portals demo — two rooms, far apart in world space, connected by a portal
//! pair you can see and walk through. Room A (red accents) sits at the
//! origin; Room B (blue accents) sits 60 units away — the portal makes them
//! read as adjacent even though nothing else about the world is.
//!
//! See `docs/portals_and_sublevels.md` for the design and
//! `crates/helio/src/scene/portals.rs` for the API this demo drives.
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Mouse drag  — look around (click to grab cursor)
//!   Escape      — release cursor / exit
//!   Walk through the glowing doorway to teleport between rooms.

mod v3_demo_common;

use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera,
    DebugDrawState, GroupMask, MaterialId, MeshId, ObjectDescriptor, PortalDescriptor, PortalPose,
    Renderer, RendererConfig, Scene,
};
use helio_default_graphs::build_default_graph;
use v3_demo_common::{box_mesh, cube_mesh, make_material, plane_mesh, point_light};

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use std::collections::HashSet;
use std::sync::Arc;

/// Half-extent of the portal's rectangular opening (matches the doorway
/// meshes built into each room).
const DOORWAY_HALF_EXTENT: glam::Vec2 = glam::Vec2::new(1.2, 2.0);

fn main() {
    env_logger::init();
    log::info!("Starting Helio Portals Demo");

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
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    teleport_flashes_remaining: u32,
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
                        .with_title("Helio — Portals Demo")
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
        renderer_config.enable_portals = true;

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

        // Room A: red accents, centred at the origin, doorway on its +Z wall.
        // Room B: blue accents, 60 units away, doorway on its -Z wall facing
        // back toward where A's doorway "should" be — the portal is what
        // actually makes the two line up, nothing else about their world
        // positions does.
        build_room(&mut renderer, glam::Vec3::ZERO, 0.0, [0.75, 0.2, 0.2, 1.0], "A");
        build_room(&mut renderer, glam::Vec3::new(60.0, 0.0, 0.0), std::f32::consts::PI, [0.2, 0.35, 0.8, 1.0], "B");

        let portal_a_pos = glam::Vec3::new(0.0, 1.6, 6.0);
        let portal_b_pos = glam::Vec3::new(60.0, 1.6, -6.0);
        let a = PortalPose::from_look_at(portal_a_pos, portal_a_pos + glam::Vec3::Z, glam::Vec3::Y);
        let b = PortalPose::from_look_at(portal_b_pos, portal_b_pos - glam::Vec3::Z, glam::Vec3::Y);
        renderer.scene_mut().add_portal(PortalDescriptor { a, b, half_extent: DOORWAY_HALF_EXTENT, user_tag: 0 });

        renderer.scene_mut().insert_light(v3_demo_common::directional_light([-0.4, -0.85, -0.3], [1.0, 0.97, 0.9], 3.0));
        renderer.scene_mut().insert_light(point_light([0.0, 4.0, 0.0], [1.0, 0.95, 0.85], 6.0, 18.0));
        renderer.scene_mut().insert_light(point_light([60.0, 4.0, 0.0], [0.85, 0.9, 1.0], 6.0, 18.0));

        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_format,
            renderer,
            last_frame: std::time::Instant::now(),
            cam_pos: glam::Vec3::new(0.0, 1.8, 0.0),
            cam_yaw: 0.0,
            cam_pitch: 0.0,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            teleport_flashes_remaining: 0,
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
        let mut forward = glam::Vec3::new(sy * cp, sp, -cy * cp);
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

        // ── Portal crossing: test BEFORE building this frame's camera ──────
        // (see `Scene::take_portal_teleport`'s doc comment for why the order
        // matters — the frame's secondary views must see the *already*
        // teleported camera, never the pre-crossing one).
        if let Some(teleport) = self.renderer.take_portal_teleport(self.cam_pos) {
            log::info!("Crossed a portal — teleported to {:?}", teleport.new_position);
            self.cam_pos = teleport.new_position;
            forward = teleport.rotation.transform_vector3(forward);
            // Re-derive yaw/pitch from the rotated forward vector so mouse
            // look continues smoothly from the new orientation (matches the
            // FPS basis this demo builds `forward` from: forward =
            // (sin(yaw)cos(pitch), sin(pitch), -cos(yaw)cos(pitch))).
            self.cam_pitch = forward.y.clamp(-1.0, 1.0).asin();
            self.cam_yaw = forward.x.atan2(-forward.z);
            self.teleport_flashes_remaining = 3;
        }
        // Drains the "reset TAA/TSR history" signal the crossing above set.
        // This demo doesn't own the temporal-history reset path itself (that
        // lives in the embedding engine — see `Renderer::portal_teleport_taa_reset`'s
        // doc comment) but still drains the flag so it doesn't accumulate.
        let _ = self.renderer.portal_teleport_taa_reset();
        if self.teleport_flashes_remaining > 0 {
            self.teleport_flashes_remaining -= 1;
        }

        let size = self.window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;

        let camera = Camera::perspective_look_at(self.cam_pos, self.cam_pos + forward, glam::Vec3::Y, std::f32::consts::FRAC_PI_4, aspect, 0.1, 300.0);

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

/// Builds a simple room: floor, back/left/right walls, a doorway gap in the
/// +Z-facing wall (rotated by `yaw_offset` so room B's doorway faces back
/// toward room A), and a colored marker cube so the two rooms are trivially
/// distinguishable through the portal.
fn build_room(renderer: &mut Renderer, center: glam::Vec3, yaw_offset: f32, accent_color: [f32; 4], label: &str) {
    let room_half = 8.0f32;
    let wall_height = 4.0f32;
    let rotation = glam::Mat4::from_rotation_y(yaw_offset);

    let floor_mat = renderer.scene_mut().insert_material(make_material([0.5, 0.5, 0.52, 1.0], 0.85, 0.0, [0.0; 3], 0.0));
    let floor_mesh = renderer.scene_mut().insert_mesh(plane_mesh([0.0, 0.0, 0.0], room_half));
    insert_static(renderer, floor_mesh, floor_mat, glam::Mat4::from_translation(center) * rotation, room_half);

    let wall_mat = renderer.scene_mut().insert_material(make_material(accent_color, 0.7, 0.0, [0.0; 3], 0.0));
    let wall_mesh = renderer.scene_mut().insert_mesh(box_mesh([0.0, 0.0, 0.0], [room_half, wall_height * 0.5, 0.2]));
    let side_wall_mesh = renderer.scene_mut().insert_mesh(box_mesh([0.0, 0.0, 0.0], [0.2, wall_height * 0.5, room_half]));

    let world = |local: glam::Vec3| glam::Mat4::from_translation(center) * rotation * glam::Mat4::from_translation(local);

    // Back wall (opposite the doorway) — solid.
    insert_static(renderer, wall_mesh, wall_mat, world(glam::Vec3::new(0.0, wall_height * 0.5, -room_half)), room_half);
    // Left/right walls — solid.
    insert_static(renderer, side_wall_mesh, wall_mat, world(glam::Vec3::new(-room_half, wall_height * 0.5, 0.0)), room_half);
    insert_static(renderer, side_wall_mesh, wall_mat, world(glam::Vec3::new(room_half, wall_height * 0.5, 0.0)), room_half);
    // Front wall (+Z), split left/right of the doorway gap (matches
    // `DOORWAY_HALF_EXTENT`).
    let gap = DOORWAY_HALF_EXTENT.x + 0.3;
    let front_segment_mesh = renderer.scene_mut().insert_mesh(box_mesh([0.0, 0.0, 0.0], [(room_half - gap) * 0.5, wall_height * 0.5, 0.2]));
    insert_static(
        renderer,
        front_segment_mesh,
        wall_mat,
        world(glam::Vec3::new(-(gap + (room_half - gap) * 0.5), wall_height * 0.5, room_half)),
        room_half,
    );
    insert_static(
        renderer,
        front_segment_mesh,
        wall_mat,
        world(glam::Vec3::new(gap + (room_half - gap) * 0.5, wall_height * 0.5, room_half)),
        room_half,
    );
    // Lintel above the doorway.
    let lintel_mesh = renderer.scene_mut().insert_mesh(box_mesh([0.0, 0.0, 0.0], [gap, (wall_height - DOORWAY_HALF_EXTENT.y * 2.0) * 0.5, 0.2]));
    insert_static(
        renderer,
        lintel_mesh,
        wall_mat,
        world(glam::Vec3::new(0.0, DOORWAY_HALF_EXTENT.y * 2.0 + (wall_height - DOORWAY_HALF_EXTENT.y * 2.0) * 0.5, room_half)),
        room_half,
    );

    // A room-identifying marker so it's obvious which side you're on.
    let marker_mesh = renderer.scene_mut().insert_mesh(cube_mesh([0.0, 0.0, 0.0], 1.0));
    let marker_mat = renderer.scene_mut().insert_material(make_material(accent_color, 0.3, 0.4, [accent_color[0] * 0.4, accent_color[1] * 0.4, accent_color[2] * 0.4], 1.0));
    insert_static(renderer, marker_mesh, marker_mat, world(glam::Vec3::new(0.0, 1.0, -4.0)), 1.0);

    log::info!("Room {label} built at {center:?}");
}

fn insert_static(renderer: &mut Renderer, mesh: MeshId, material: MaterialId, transform: glam::Mat4, radius: f32) {
    let _ = renderer
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
        .expect("insert static room object");
}
