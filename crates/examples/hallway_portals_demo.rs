//! Hallway portals demo — the `indoor_corridor` hallway with a portal pair
//! at *each* end instead of solid end walls, connected so the hallway loops
//! on itself: walk out the far end and you re-enter from the near end still
//! heading the same direction (and vice versa). A textbook non-Euclidean
//! "impossible corridor" — see docs/portals_and_sublevels.md.
//!
//! The two portal pairs share the exact same corridor geometry seen from
//! each end, which is what sells the loop: looking through either doorway
//! shows *more hallway*, because that's genuinely what's there.
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Mouse drag  — look around (click to grab cursor)
//!   Escape      — release cursor / exit
//!   Walk out either end of the hallway to loop around.

mod v3_demo_common;

use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera,
    DebugDrawState, LightId, PortalDescriptor, PortalPose, Renderer, RendererConfig, Scene,
};
use helio_default_graphs::build_default_graph;
use v3_demo_common::{box_mesh, make_material, point_light, spot_light, directional_light};

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use std::collections::HashSet;
use std::sync::Arc;

/// Corridor half-length along Z (matches `indoor_corridor`: 36 m long, ends
/// at `-HALF_LENGTH`/`+HALF_LENGTH`).
const HALF_LENGTH: f32 = 18.0;
/// Portal opening half-extent (local X = corridor width axis, local Y =
/// height) — slightly inset from the full 4 m × 3 m cross-section so a
/// visible frame remains, matching `portals_demo`'s doorway style.
const DOORWAY_HALF_EXTENT: glam::Vec2 = glam::Vec2::new(1.8, 1.35);

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
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

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
                        .with_title("Helio – Hallway Portals (Impossible Corridor)")
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
        let format = caps.formats.iter().find(|f| f.is_srgb()).copied().unwrap_or(caps.formats[0]);
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

        let mut config = RendererConfig::new(size.width, size.height, format);
        // The one flag that turns portal rendering on — see
        // docs/portals_and_sublevels.md's "zero overhead when absent"
        // guarantee for why this defaults to `false`.
        config.enable_portals = true;

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

        let mat = renderer.scene_mut().insert_material(make_material(
            [0.72, 0.72, 0.75, 1.0],
            0.8,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));
        let frame_mat = renderer.scene_mut().insert_material(make_material(
            [0.15, 0.55, 0.95, 1.0],
            0.4,
            0.3,
            [0.05, 0.2, 0.4],
            0.6,
        )        );

        // Corridor: 4 m wide (X), 3 m tall (Y), 36 m long (Z: -18..+18) —
        // identical to `indoor_corridor`, minus the solid end walls (a
        // portal-framed opening sits there instead).
        let floor = renderer.scene_mut().insert_mesh(box_mesh([0.0, 0.0, 0.0], [2.0, 0.02, HALF_LENGTH]));
        let ceiling = renderer.scene_mut().insert_mesh(box_mesh([0.0, 0.0, 0.0], [2.0, 0.02, HALF_LENGTH]));
        let wall_l = renderer.scene_mut().insert_mesh(box_mesh([0.0, 0.0, 0.0], [0.02, 1.5, HALF_LENGTH]));
        let wall_r = renderer.scene_mut().insert_mesh(box_mesh([0.0, 0.0, 0.0], [0.02, 1.5, HALF_LENGTH]));
        let sconce_l = renderer.scene_mut().insert_mesh(box_mesh([0.0, 0.0, 0.0], [0.12, 0.08, 0.25]));
        let sconce_r = renderer.scene_mut().insert_mesh(box_mesh([0.0, 0.0, 0.0], [0.12, 0.08, 0.25]));

        v3_demo_common::insert_object(&mut renderer, floor, mat, glam::Mat4::IDENTITY, HALF_LENGTH).ok();
        v3_demo_common::insert_object(&mut renderer, ceiling, mat, glam::Mat4::from_translation(glam::Vec3::new(0.0, 3.0, 0.0)), HALF_LENGTH).ok();
        v3_demo_common::insert_object(&mut renderer, wall_l, mat, glam::Mat4::from_translation(glam::Vec3::new(-2.0, 1.5, 0.0)), HALF_LENGTH).ok();
        v3_demo_common::insert_object(&mut renderer, wall_r, mat, glam::Mat4::from_translation(glam::Vec3::new(2.0, 1.5, 0.0)), HALF_LENGTH).ok();
        v3_demo_common::insert_object(&mut renderer, sconce_l, mat, glam::Mat4::from_translation(glam::Vec3::new(-1.85, 1.8, 0.0)), 0.3).ok();
        v3_demo_common::insert_object(&mut renderer, sconce_r, mat, glam::Mat4::from_translation(glam::Vec3::new(1.85, 1.8, 0.0)), 0.3).ok();

        // Door frames around each portal opening (top lintel + two side
        // jambs), purely cosmetic — makes the portal boundary legible
        // instead of the hallway just trailing off into nothing.
        build_doorway_frame(&mut renderer, frame_mat, -HALF_LENGTH);
        build_doorway_frame(&mut renderer, frame_mat, HALF_LENGTH);

        // ── The loop: two portal pairs, one per direction of travel ────────
        //
        // Both poses in a pair share the *same* world-space orientation
        // (only their position differs), which makes `pair_map` a pure
        // translation — crossing one end deposits you at the other end
        // still facing the direction you were already walking, so forward
        // motion just keeps cycling through the same 36 m hallway.
        //
        // Far-end pair: crossing z=-18 heading further -Z re-enters at
        // z=+18, still heading -Z (i.e. walking back into the hallway from
        // the near end).
        //
        // from_look_at convention: portal center world = (eye.x, -eye.y, dot(f,eye))
        // where f = normalize(look_at - eye). For f = -Z: dot(f,eye) = -eye.z,
        // so z is NEGATED. For f = +Z: dot(f,eye) = eye.z, z is UNCHANGED.
        // y is ALWAYS negated (dot(u,eye) = eye.y → w_axis.y = -eye.y).
        // So to place a portal at (wx, wy, wz) with forward f:
        //   from_look_at((wx, -wy, wz), ...)  when f = +Z
        //   from_look_at((wx, -wy, -wz), ...) when f = -Z
        //
        // Pair 1: A at z=-18 forward=-Z, B at z=+18 forward=-Z
        let far_eye = glam::Vec3::new(0.0, -1.5, HALF_LENGTH); // -(-18) = 18
        let far_outward = PortalPose::from_look_at(
            far_eye,
            far_eye + glam::Vec3::new(0.0, 0.0, -1.0),
            glam::Vec3::Y,
        );
        let far_dest_eye = glam::Vec3::new(0.0, -1.5, -HALF_LENGTH); // -(18) = -18
        let far_destination = PortalPose::from_look_at(
            far_dest_eye,
            far_dest_eye + glam::Vec3::new(0.0, 0.0, -1.0),
            glam::Vec3::Y,
        );
        renderer.scene_mut().add_portal(PortalDescriptor {
            a: far_outward,
            b: far_destination,
            half_extent: DOORWAY_HALF_EXTENT,
            user_tag: 0,
        });

        // Pair 2: A at z=+18 forward=+Z, B at z=-18 forward=+Z
        let near_outward = PortalPose::from_look_at(
            glam::Vec3::new(0.0, -1.5, HALF_LENGTH),
            glam::Vec3::new(0.0, -1.5, HALF_LENGTH + 1.0),
            glam::Vec3::Y,
        );
        let near_destination = PortalPose::from_look_at(
            glam::Vec3::new(0.0, -1.5, -HALF_LENGTH),
            glam::Vec3::new(0.0, -1.5, -HALF_LENGTH + 1.0),
            glam::Vec3::Y,
        );
        renderer.scene_mut().add_portal(PortalDescriptor {
            a: near_outward,
            b: near_destination,
            half_extent: DOORWAY_HALF_EXTENT,
            user_tag: 0,
        });

        let mut light_ids = Vec::new();
        for &z in &[-14.0f32, -7.0, 0.0, 7.0, 14.0] {
            light_ids.push(renderer.scene_mut().insert_light(spot_light(
                [0.0, 2.88, z],
                [0.0, -1.0, 0.0],
                [0.9, 0.95, 1.0],
                3.5,
                9.0,
                1.22,
                1.48,
            )));
        }
        light_ids.push(renderer.scene_mut().insert_light(point_light([-1.7, 1.85, 0.0], [1.0, 0.65, 0.3], 2.0, 4.5)));
        light_ids.push(renderer.scene_mut().insert_light(point_light([1.7, 1.85, 0.0], [1.0, 0.65, 0.3], 2.0, 4.5)));
        // Portal-frame accent lights so each doorway reads clearly from a
        // distance.
        light_ids.push(renderer.scene_mut().insert_light(point_light([0.0, 1.5, -HALF_LENGTH + 1.5], [0.2, 0.6, 1.0], 3.0, 6.0)));
        light_ids.push(renderer.scene_mut().insert_light(point_light([0.0, 1.5, HALF_LENGTH - 1.5], [0.2, 0.6, 1.0], 3.0, 6.0)));
        renderer.set_ambient([0.85, 0.9, 1.0], 0.04);
        renderer.set_clear_color([0.0, 0.0, 0.0, 1.0]);

        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_format: format,
            renderer,
            last_frame: std::time::Instant::now(),
            cam_pos: glam::Vec3::new(0.0, 1.6, 16.0),
            cam_yaw: std::f32::consts::PI,
            cam_pitch: 0.0,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            _light_ids: light_ids,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
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
        const SPEED: f32 = 5.0;
        const SENS: f32 = 0.002;

        self.cam_yaw += self.mouse_delta.0 * SENS;
        self.cam_pitch = (self.cam_pitch - self.mouse_delta.1 * SENS).clamp(-1.4, 1.4);
        self.mouse_delta = (0.0, 0.0);

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let mut forward = glam::Vec3::new(sy * cp, sp, -cy * cp);
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

        // Portal crossing — must run *before* building this frame's camera;
        // see `Scene::take_portal_teleport`'s doc comment.
        if let Some(teleport) = self.renderer.take_portal_teleport(self.cam_pos) {
            log::info!("Looped through the hallway — now at {:?}", teleport.new_position);
            self.cam_pos = teleport.new_position;
            forward = teleport.rotation.transform_vector3(forward);
            self.cam_pitch = forward.y.clamp(-1.0, 1.0).asin();
            self.cam_yaw = forward.x.atan2(-forward.z);
        }
        let _ = self.renderer.portal_teleport_taa_reset();

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
            wgpu::CurrentSurfaceTexture::Success(texture) | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            _ => return,
        };
        let view = output.texture.create_view(&Default::default());

        if let Err(e) = self.renderer.render(&camera, &view) {
            log::error!("Render: {:?}", e);
        }
        self.queue.present(output);
    }
}

/// A thin lintel + two jambs framing the portal opening at `z`, sized around
/// `DOORWAY_HALF_EXTENT`.
fn build_doorway_frame(renderer: &mut Renderer, material: helio::MaterialId, z: f32) {
    let jamb_mesh = renderer.scene_mut().insert_mesh(box_mesh([0.0, 0.0, 0.0], [0.1, 1.5, 0.1]));
    let lintel_mesh = renderer.scene_mut().insert_mesh(box_mesh([0.0, 0.0, 0.0], [DOORWAY_HALF_EXTENT.x + 0.1, 0.1, 0.1]));

    v3_demo_common::insert_object(
        renderer,
        jamb_mesh,
        material,
        glam::Mat4::from_translation(glam::Vec3::new(-DOORWAY_HALF_EXTENT.x, 1.5, z)),
        1.5,
    )
    .ok();
    v3_demo_common::insert_object(
        renderer,
        jamb_mesh,
        material,
        glam::Mat4::from_translation(glam::Vec3::new(DOORWAY_HALF_EXTENT.x, 1.5, z)),
        1.5,
    )
    .ok();
    v3_demo_common::insert_object(
        renderer,
        lintel_mesh,
        material,
        glam::Mat4::from_translation(glam::Vec3::new(0.0, DOORWAY_HALF_EXTENT.y * 2.0, z)),
        DOORWAY_HALF_EXTENT.x + 0.1,
    )
    .ok();
}
