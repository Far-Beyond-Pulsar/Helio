//! Portal cube — a sealed room with a doorway-shaped portal in the center of
//! each of its 6 walls, each one reflecting the *same real room* back at
//! itself. No manually-authored "copies" anywhere in this file: every
//! reflection you see — including the second, third bounce receding into
//! each doorway — comes entirely from `helio::Scene::add_portal` and the
//! engine's own portal-chain composition (`helio-pass-portal-cull` /
//! `helio-pass-portal-instances`). This is the automatic-recursion
//! generalization of `infinite_tunnel`'s single hand-placed corridor: a
//! portal here pairs its real doorway with the pose *at the opposite wall,
//! facing the same direction* — a real, physical position with real content
//! already there (the room's own opposite side), not a hidden buried copy.
//! Because the engine composes portals into chains automatically, each
//! doorway shows the room repeating away from you for a few bounces, and —
//! since every portal's "far side" is itself a room with its own 6 portals —
//! standing near a corner you can see one doorway's reflection through
//! another's.
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Mouse drag  — look around (click to grab cursor)
//!   Tab         — toggle editor mode (on by default): shows a checkerboard
//!                 over each portal opening so you can see where it is.
//!   Escape      — release cursor / exit

mod v3_demo_common;

use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera,
    DebugDrawState, GroupMask, LightId, ObjectDescriptor, PortalDescriptor, PortalId, Renderer,
    RendererConfig, Scene, SceneActor,
};
use helio_default_graphs::build_default_graph;
use v3_demo_common::{box_mesh, make_material, point_light};

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use glam::{Mat4, Vec2, Vec3};
use std::collections::HashSet;
use std::sync::Arc;

/// Room interior half-extent — the room spans [-HALF_SIZE, HALF_SIZE] on
/// every axis, so it's a 2*HALF_SIZE cube.
const HALF_SIZE: f32 = 6.0;
/// Doorway half-width (local X) / half-height (local Y) in each wall's own
/// face-local frame — also the portal clip opening's half-extent.
const DOOR_HALF_W: f32 = 1.6;
const DOOR_HALF_H: f32 = 2.3;
/// Wall panel half-thickness.
const WALL_T: f32 = 0.15;
/// Far plane — generous enough that a few chain bounces (each one roughly
/// 2*HALF_SIZE farther away) all stay comfortably inside it.
const FAR_PLANE: f32 = 300.0;

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

    cam_pos: Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    _portal_ids: Vec<PortalId>,
    _light_ids: Vec<LightId>,

    /// Debug-only: when `CUBE_SCREENSHOT` is set, counts frames so a single
    /// PNG can be captured after the scene has settled, then the process exits.
    frame_count: u32,
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
                        .with_title("Helio – Portal Cube")
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

        let mut config = RendererConfig::new(size.width, size.height, format);
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

        // ── Materials ───────────────────────────────────────────────────────
        let wall_mat = renderer.scene_mut().insert_material(make_material(
            [0.75, 0.75, 0.78, 1.0], 0.75, 0.0, [0.0, 0.0, 0.0], 0.0,
        ));
        let frame_mat = renderer.scene_mut().insert_material(make_material(
            [0.3, 0.9, 1.0, 1.0], 0.4, 0.0, [0.2, 0.85, 1.0], 2.5,
        ));

        // Single shared unit box (half-extent 1 on every axis) — every wall
        // panel and frame piece is this same mesh, scaled/rotated/positioned
        // per instance via its own transform (see `insert_wall_face` below).
        let unit_mesh = renderer.scene_mut().insert_actor(SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))).as_mesh().unwrap();

        // ── The room: one wall per axis direction, each with a centered
        // doorway. `up_hint` just needs to not be parallel to `normal` — Y
        // for the four side walls, Z for the floor/ceiling (where normal
        // itself is ±Y). The orthonormalized (right, up, normal) triple this
        // produces is reused directly as the portal's own pose basis below,
        // so the doorway hole and the portal's clip window always agree.
        let faces: [(Vec3, Vec3); 6] = [
            (Vec3::X, Vec3::Y),
            (Vec3::NEG_X, Vec3::Y),
            (Vec3::Y, Vec3::Z),
            (Vec3::NEG_Y, Vec3::Z),
            (Vec3::Z, Vec3::Y),
            (Vec3::NEG_Z, Vec3::Y),
        ];

        let mut portal_ids = Vec::new();
        for &(normal, up_hint) in &faces {
            let right = up_hint.cross(normal).normalize();
            let up = normal.cross(right).normalize();

            insert_wall_face(&mut renderer, unit_mesh, wall_mat, frame_mat, normal, right, up);

            // Pair this doorway with the pose at the *opposite* wall, facing
            // the *same* direction as this one (not that wall's own outward
            // normal) — that's what makes the map a pure "keep going
            // straight" translation by the room's full size, using the
            // opposite wall's real content, not a fabricated stand-in. See
            // the module doc for why this is the whole trick.
            let a = helio::portal_pose_facing(normal * HALF_SIZE, normal, up);
            let b = helio::portal_pose_facing(-normal * HALF_SIZE, normal, up);
            let portal = renderer
                .scene_mut()
                .add_portal(PortalDescriptor { a, b, half_extent: Vec2::new(DOOR_HALF_W, DOOR_HALF_H) })
                .expect("add_portal");
            portal_ids.push(portal);
        }

        // ── A light near the center so every wall reads, plus one per
        // doorway direction so the receding reflections don't go flat black.
        let mut light_ids = Vec::new();
        light_ids.push(
            renderer.scene_mut().insert_actor(SceneActor::light(point_light([0.0, HALF_SIZE * 0.85, 0.0], [1.0, 0.98, 0.92], 4.0, HALF_SIZE * 1.8)))
                .as_light().unwrap(),
        );
        // Just 2 more (not one per face — 7 overlapping light-range gizmos
        // in editor mode turned into unreadable clutter) at opposite
        // corners, enough to break up the single center light's flatness.
        for &pos in &[Vec3::new(3.5, 3.0, 3.5), Vec3::new(-3.5, -3.0, -3.5)] {
            light_ids.push(
                renderer.scene_mut().insert_actor(SceneActor::light(point_light([pos.x, pos.y, pos.z], [0.85, 0.92, 1.0], 2.0, HALF_SIZE)))
                    .as_light().unwrap(),
            );
        }

        // Deferred lighting shades every pixel — including portal
        // duplicates — by real distance to the scene's real lights, and a
        // duplicated reflection can be mapped a full room-length or more
        // away from all of them. Ambient is position-independent, so unlike
        // point lights it reaches reflections just as well as the real
        // room; pushed up well past a typical scene's ambient so the walls
        // read clearly at every reflection depth instead of fading to black
        // a couple of bounces in.
        renderer.set_ambient([0.6, 0.65, 0.75], 0.35);
        renderer.set_clear_color([0.0, 0.0, 0.0, 1.0]);

        // Editor mode (Tab to toggle) starts on so the portal-opening
        // checkerboard — otherwise invisible, see PortalEditorOverlayPass —
        // is visible by default; press Tab to see the fully seamless game-
        // mode look.
        renderer.set_editor_mode(std::env::var("CUBE_NO_EDITOR").is_err());

        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_format: format,
            renderer,
            last_frame: std::time::Instant::now(),
            // Deliberately off the centerline of every doorway (near a
            // corner, not on any face's own axis) — a portal pair here is a
            // pure translation with no zoom, so looking exactly through a
            // doorway's dead center only ever shows the identically-sized
            // doorway hole on the opposite wall, recursively (a nested
            // picture-frame effect) — solid wall content only comes into
            // view off that centerline, same as it would with two real
            // rooms and a real window between them.
            cam_pos: if std::env::var("CUBE_CLOSEUP").is_ok() { Vec3::new(1.5, 1.0, 3.5) } else { Vec3::new(4.0, 3.0, 4.0) },
            cam_yaw: if std::env::var("CUBE_CLOSEUP").is_ok() { -2.60 } else { -0.785 },
            cam_pitch: if std::env::var("CUBE_CLOSEUP").is_ok() { -0.33 } else { -0.488 },
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            _portal_ids: portal_ids,
            _light_ids: light_ids,
            frame_count: 0,
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
            WindowEvent::KeyboardInput {
                event: KeyEvent { state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::Tab), repeat: false, .. },
                ..
            } => {
                let enabled = !state.renderer.is_editor_mode();
                state.renderer.set_editor_mode(enabled);
                log::info!("[portal_cube] editor mode: {}", enabled);
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent { state: ks, physical_key: PhysicalKey::Code(key), .. },
                ..
            } => match ks {
                ElementState::Pressed => { state.keys.insert(key); }
                ElementState::Released => { state.keys.remove(&key); }
            },
            WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => {
                if !state.cursor_grabbed {
                    let ok = state.window.set_cursor_grab(CursorGrabMode::Confined)
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
        let forward = Vec3::new(sy * cp, sp, -cy * cp);
        let right = Vec3::new(cy, 0.0, sy);

        if self.keys.contains(&KeyCode::KeyW) { self.cam_pos += forward * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyS) { self.cam_pos -= forward * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyA) { self.cam_pos -= right * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyD) { self.cam_pos += right * SPEED * dt; }
        if self.keys.contains(&KeyCode::Space) { self.cam_pos += Vec3::Y * SPEED * dt; }
        if self.keys.contains(&KeyCode::ShiftLeft) { self.cam_pos -= Vec3::Y * SPEED * dt; }

        let size = self.window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;

        let camera = Camera::perspective_look_at(
            self.cam_pos, self.cam_pos + forward, Vec3::Y,
            std::f32::consts::FRAC_PI_4, aspect, 0.1, FAR_PLANE,
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

        // ── Debug screenshot path — see infinite_tunnel.rs for the full
        // rationale (no way to drive a native window interactively here).
        self.frame_count += 1;
        if let Ok(path) = std::env::var("CUBE_SCREENSHOT") {
            const CAPTURE_AT_FRAME: u32 = 90;
            if self.frame_count == CAPTURE_AT_FRAME {
                let offscreen = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Screenshot Offscreen"),
                    size: wgpu::Extent3d { width: size.width, height: size.height, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: self.surface_format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                let offscreen_view = offscreen.create_view(&Default::default());
                if let Err(e) = self.renderer.render(&camera, &offscreen_view) {
                    log::error!("Screenshot render: {:?}", e);
                }
                capture_screenshot(&self.device, &self.queue, &offscreen, self.surface_format, &path);
                self.queue.present(output);
                std::process::exit(0);
            }
        }

        self.queue.present(output);
    }
}

/// Inserts one wall's 4 panels (top/bottom/left/right around a centered
/// doorway) plus an emissive frame collar around the opening, all using the
/// shared unit box mesh scaled per panel via its own transform. `(right,
/// up, normal)` must be orthonormal — see the call site for how that's
/// built from each face's `up_hint`.
fn insert_wall_face(
    renderer: &mut Renderer,
    unit_mesh: helio::MeshId,
    wall_mat: helio::MaterialId,
    frame_mat: helio::MaterialId,
    normal: Vec3,
    right: Vec3,
    up: Vec3,
) {
    let face_center = normal * HALF_SIZE;
    let mut insert_box = |material: helio::MaterialId, center_right: f32, center_up: f32, center_normal: f32, half_right: f32, half_up: f32, half_normal: f32| {
        let center = face_center + right * center_right + up * center_up + normal * center_normal;
        let transform = Mat4::from_cols(
            (right * half_right).extend(0.0),
            (up * half_up).extend(0.0),
            (normal * half_normal).extend(0.0),
            center.extend(1.0),
        );
        let radius = (half_right * half_right + half_up * half_up + half_normal * half_normal).sqrt();
        // Deliberately *not* INSTANCE_FLAG_ALWAYS_VISIBLE — unlike
        // infinite_tunnel's single central corridor segment (world-space,
        // no portal chain ever needs to reconsider it), every wall panel
        // here is real content that portal chains actively frustum-test
        // through their own composed transforms. ALWAYS_VISIBLE would make
        // that test a no-op, so *every* chain thinks *every* panel is in
        // view regardless of where it actually maps to — wildly
        // overselecting and blowing straight through the cull pass's
        // per-group capacity.
        let _ = renderer.scene_mut().insert_actor(SceneActor::object(ObjectDescriptor {
            mesh: unit_mesh,
            material,
            transform,
            bounds: [center.x, center.y, center.z, radius],
            flags: 0,
            groups: GroupMask::NONE,
            movability: None,
            user_tag: 0,
        }));
    };

    // Top / bottom panels span the full width; left / right panels fill the
    // remaining height beside the doorway.
    let vert_half = (HALF_SIZE - DOOR_HALF_H) / 2.0;
    let vert_center = DOOR_HALF_H + vert_half;
    insert_box(wall_mat, 0.0, vert_center, 0.0, HALF_SIZE, vert_half, WALL_T);
    insert_box(wall_mat, 0.0, -vert_center, 0.0, HALF_SIZE, vert_half, WALL_T);
    let horiz_half = (HALF_SIZE - DOOR_HALF_W) / 2.0;
    let horiz_center = DOOR_HALF_W + horiz_half;
    insert_box(wall_mat, horiz_center, 0.0, 0.0, horiz_half, DOOR_HALF_H, WALL_T);
    insert_box(wall_mat, -horiz_center, 0.0, 0.0, horiz_half, DOOR_HALF_H, WALL_T);

    // Emissive frame collar right at the doorway's edge, thin boxes just
    // outside the opening — same purely-decorative role as
    // infinite_tunnel's own portal frames.
    let ft = 0.08;
    insert_box(frame_mat, 0.0, DOOR_HALF_H + ft, ft, DOOR_HALF_W + ft, ft, ft);
    insert_box(frame_mat, 0.0, -(DOOR_HALF_H + ft), ft, DOOR_HALF_W + ft, ft, ft);
    insert_box(frame_mat, DOOR_HALF_W + ft, 0.0, ft, ft, DOOR_HALF_H + ft, ft);
    insert_box(frame_mat, -(DOOR_HALF_W + ft), 0.0, ft, ft, DOOR_HALF_H + ft, ft);
}

/// Copies `texture` (must have been created/configured with `COPY_SRC`) back
/// to the CPU and writes it out as a PNG. Assumes an 8-bit-per-channel RGBA
/// or BGRA surface format (true for every format the surface capability query
/// in this example can select).
fn capture_screenshot(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    format: wgpu::TextureFormat,
    path: &str,
) {
    let width = texture.width();
    let height = texture.height();
    let bytes_per_pixel = 4u32;
    let unpadded_bytes_per_row = width * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Screenshot Readback"),
        size: (padded_bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Screenshot Encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
    queue.submit(Some(encoder.finish()));

    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map screenshot buffer"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range().expect("get_mapped_range");

    let is_bgra = matches!(
        format,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    );
    let mut pixels = vec![0u8; (width * height * bytes_per_pixel) as usize];
    for row in 0..height {
        let src_start = (row * padded_bytes_per_row) as usize;
        let src_row = &data[src_start..src_start + unpadded_bytes_per_row as usize];
        let dst_start = (row * unpadded_bytes_per_row) as usize;
        let dst_row = &mut pixels[dst_start..dst_start + unpadded_bytes_per_row as usize];
        dst_row.copy_from_slice(src_row);
        if is_bgra {
            for px in dst_row.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
        }
    }
    drop(data);
    buffer.unmap();

    image::save_buffer(path, &pixels, width, height, image::ColorType::Rgba8)
        .expect("save screenshot png");
    log::info!("[Screenshot] wrote {path} ({width}x{height})");
}
