//! Portal rooms — a cube with *no walls of its own at all*. Each of its 6
//! faces is nothing but a portal, edge to edge, no doorway cut into a wall
//! and no frame or border around it — the face itself is the entire portal
//! surface. Stand in the middle and every face shows a different scene
//! filling it completely — an ember-lit room to the east, a glacier to the
//! west, a room overhead, a room below — with nothing marking where the
//! "wall" is, because there isn't one. That's deliberate: without a doorway
//! shape or a frame telling you "this is the portal, right here", it's not
//! obvious at a glance which face is solid-looking-but-isn't, which makes
//! the illusion more disorienting (in a good way) than `portal_cube`'s
//! framed doorways. None of it is faked: each face is a real
//! `helio::Scene::add_portal` pairing that whole face with the real entrance
//! of a real room built somewhere else in world space, and the engine's own
//! portal pipeline (`helio-pass-portal-cull` / `helio-pass-portal-instances`)
//! does the rest — the only difference from `portal_cube` is that the
//! portal's `half_extent` covers the *entire* face instead of a doorway
//! inset into a wall, and that wall is never built in the first place.
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

/// Hub half-extent — there's no hub geometry at all (see the module doc):
/// this is purely the position of each face's portal plane and the
/// half-extent of its clip window, so the portal covers the *entire* face,
/// corner to corner.
const HUB_HALF_SIZE: f32 = 6.0;
/// Side-room wall panel half-thickness.
const WALL_T: f32 = 0.15;

/// Each side room's own half-extent — bigger than the hub, so stepping
/// "through" a doorway visually reads as entering a larger, different space.
const ROOM_HALF_SIZE: f32 = 8.0;
/// Distance from the hub's center to each side room's center, measured along
/// that room's own axis. Rooms hang off the hub like the arms of a plus
/// sign, one per axis direction, so — given `ROOM_SPACING - ROOM_HALF_SIZE >
/// HUB_HALF_SIZE` — no two rooms' geometry ever overlaps in world space.
const ROOM_SPACING: f32 = 40.0;

/// Far plane — generous enough to cover the farthest room wall
/// (ROOM_SPACING + ROOM_HALF_SIZE) plus real flying-around margin.
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

    /// Debug-only: when `ROOMS_SCREENSHOT` is set, counts frames so a single
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
                        .with_title("Helio – Portal Rooms")
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

        // Single shared unit box (half-extent 1 on every axis) — every side
        // room's shell panel and floating accent prop in this scene is this
        // same mesh, scaled/positioned per instance via its own transform
        // (see `insert_room_shell` below). The hub itself has no geometry —
        // see the module doc.
        let unit_mesh = renderer.scene_mut().insert_actor(SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))).as_mesh().unwrap();

        // ── The hub: one full-face portal per axis direction, no wall, no
        // doorway cutout, no frame. `up_hint` just needs to not be parallel
        // to `normal` — Y for the four side faces, Z for the top/bottom
        // (where normal itself is ±Y). The orthonormalized (right, up,
        // normal) triple this produces is the portal's own pose basis below.
        struct RoomTheme {
            name: &'static str,
            wall_color: [f32; 4],
            accent: [f32; 3],
        }
        let faces: [(Vec3, Vec3, RoomTheme); 6] = [
            (Vec3::X, Vec3::Y, RoomTheme { name: "Ember", wall_color: [0.55, 0.12, 0.08, 1.0], accent: [1.0, 0.35, 0.1] }),
            (Vec3::NEG_X, Vec3::Y, RoomTheme { name: "Verdant", wall_color: [0.08, 0.4, 0.14, 1.0], accent: [0.25, 1.0, 0.35] }),
            (Vec3::Y, Vec3::Z, RoomTheme { name: "Solar", wall_color: [0.55, 0.45, 0.05, 1.0], accent: [1.0, 0.85, 0.15] }),
            (Vec3::NEG_Y, Vec3::Z, RoomTheme { name: "Abyssal", wall_color: [0.05, 0.1, 0.4, 1.0], accent: [0.2, 0.4, 1.0] }),
            (Vec3::Z, Vec3::Y, RoomTheme { name: "Orchid", wall_color: [0.42, 0.08, 0.48, 1.0], accent: [0.9, 0.25, 1.0] }),
            (Vec3::NEG_Z, Vec3::Y, RoomTheme { name: "Glacier", wall_color: [0.08, 0.4, 0.45, 1.0], accent: [0.2, 0.95, 1.0] }),
        ];

        let mut portal_ids = Vec::new();
        let mut light_ids = Vec::new();
        for (normal, up_hint, theme) in &faces {
            let normal = *normal;
            let up_hint = *up_hint;
            let right = up_hint.cross(normal).normalize();
            let up = normal.cross(right).normalize();

            // The destination room for this face sits `ROOM_SPACING` units
            // out along the *opposite* of this face's outward normal — i.e.
            // behind the face from an outside viewer's point of view. This
            // is what makes the portal correct for someone standing outside
            // the cube looking in: real content needs to sit on the far
            // side of the face plane from wherever the viewer is, and the
            // viewer here is outside (in front of the face, along +normal),
            // so the room goes on the -normal side. See the `a`/`b` pose
            // comment below for why the room's own placement direction has
            // to agree with the portal's `forward` axis, not just its
            // `half_extent`.
            let room_center = -normal * ROOM_SPACING;
            let room_wall_mat = renderer.scene_mut().insert_material(make_material(
                theme.wall_color, 0.85, 0.0, [0.0, 0.0, 0.0], 0.0,
            ));
            let room_accent_mat = renderer.scene_mut().insert_material(make_material(
                [theme.accent[0], theme.accent[1], theme.accent[2], 1.0], 0.3, 0.0, theme.accent, 3.0,
            ));
            // Leave the wall facing back toward the hub (+normal, since the
            // room now sits on the -normal side) open — that's the room's
            // real "entrance", the same real surface the portal's far pose
            // sits at, so there's real geometry (floor, ceiling, far wall,
            // side walls) waiting right where the doorway leads.
            insert_room_shell(&mut renderer, unit_mesh, room_wall_mat, room_center, ROOM_HALF_SIZE, WALL_T, normal);
            // A floating accent prop at the room's center so each
            // destination reads as visually distinct at a glance, plus a
            // matching light so the room isn't lit solely by the hub's
            // distant, unrelated lights.
            insert_box_panel(&mut renderer, unit_mesh, room_accent_mat, room_center, Vec3::splat(1.4));
            light_ids.push(
                renderer.scene_mut()
                    .insert_actor(SceneActor::light(point_light(room_center.into(), theme.accent, 3.5, ROOM_HALF_SIZE * 1.8)))
                    .as_light().unwrap(),
            );
            log::info!("[portal_rooms] {} room centered at {:?}", theme.name, room_center);

            // Pair the hub's whole face with the real entrance of this room
            // — the pose at the room's near wall, facing *into* the cube
            // (`-normal`, not the face's own outward normal) — a pure "keep
            // going straight" translation using the room's own real content.
            // The forward direction here is what an outside viewer actually
            // looks along to see through the face (standing in front of it,
            // along +normal, looking toward -normal), so real content has
            // to be placed — and clipped — on that same -normal side of the
            // face plane, which is exactly where `room_center` was put
            // above. Getting `forward` and the room's placement direction
            // to agree is the entire difference from `portal_cube`, whose
            // doorways are only ever viewed from *inside* looking outward
            // (`+normal`) — see that demo's module doc for the base trick
            // this still builds on. `half_extent` covers the entire face —
            // full corner-to-corner coverage, no doorway inset — which is
            // what makes the face itself *be* the portal instead of a hole
            // cut into a wall.
            let forward = -normal;
            let a = helio::portal_pose_facing(normal * HUB_HALF_SIZE, forward, up);
            let b = helio::portal_pose_facing(room_center + normal * ROOM_HALF_SIZE, forward, up);
            let portal = renderer
                .scene_mut()
                .add_portal(PortalDescriptor { a, b, half_extent: Vec2::new(HUB_HALF_SIZE, HUB_HALF_SIZE) })
                .expect("add_portal");
            portal_ids.push(portal);
        }

        // ── A light near the hub's center so its own walls read clearly.
        light_ids.push(
            renderer.scene_mut().insert_actor(SceneActor::light(point_light([0.0, HUB_HALF_SIZE * 0.85, 0.0], [1.0, 0.98, 0.92], 4.0, HUB_HALF_SIZE * 1.8)))
                .as_light().unwrap(),
        );

        // Deferred lighting shades every pixel — including portal
        // duplicates — by real distance to the scene's real lights. Each
        // room has its own themed light for that reason, but ambient is
        // still pushed up well past a typical scene's default so every room
        // reads clearly regardless of camera position, same as `portal_cube`.
        renderer.set_ambient([0.5, 0.52, 0.58], 0.3);
        renderer.set_clear_color([0.0, 0.0, 0.0, 1.0]);

        // Editor mode (Tab to toggle) starts on so the portal-opening
        // checkerboard — otherwise invisible, see PortalEditorOverlayPass —
        // is visible by default; press Tab to see the fully seamless game-
        // mode look.
        renderer.set_editor_mode(std::env::var("ROOMS_NO_EDITOR").is_err());

        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_format: format,
            renderer,
            last_frame: std::time::Instant::now(),
            // Off the centerline of every doorway, near a corner, so
            // several different rooms' doorways are all in view — and none
            // of them is seen exactly head-on, which for a pure-translation
            // portal pair would just show the identically-shaped doorway
            // hole again (see `portal_cube`'s own comment on this).
            cam_pos: Vec3::new(4.0, 3.0, 4.0),
            cam_yaw: -0.785,
            cam_pitch: -0.488,
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
                log::info!("[portal_rooms] editor mode: {}", enabled);
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
        if let Ok(path) = std::env::var("ROOMS_SCREENSHOT") {
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

/// Inserts a single axis-aligned box panel (the shared unit mesh scaled to
/// `half_extent` and moved to `center`) — the shared building block for
/// every side room's shell walls plus its floating accent prop.
fn insert_box_panel(
    renderer: &mut Renderer,
    unit_mesh: helio::MeshId,
    material: helio::MaterialId,
    center: Vec3,
    half_extent: Vec3,
) {
    let transform = Mat4::from_cols(
        (Vec3::X * half_extent.x).extend(0.0),
        (Vec3::Y * half_extent.y).extend(0.0),
        (Vec3::Z * half_extent.z).extend(0.0),
        center.extend(1.0),
    );
    let radius = half_extent.length();
    let _ = renderer.scene_mut().insert_actor(SceneActor::object(ObjectDescriptor {
        mesh: unit_mesh,
        material,
        transform,
        bounds: [center.x, center.y, center.z, radius],
        // Not ALWAYS_VISIBLE for the same reason as `insert_wall_face`: this
        // content is only ever seen mapped through the portal that pairs
        // with this room, and the cull pass needs to actually test it.
        flags: 0,
        groups: GroupMask::NONE,
        movability: None,
        user_tag: 0,
    }));
}

/// Inserts a side room's shell: a `half_size`-cube built from 6 axis-aligned
/// wall panels of thickness `wall_t`, skipping whichever face's outward
/// normal matches `open_normal` (within tolerance) — that's the room's real
/// "entrance", the exact surface a portal's far pose sits at, so it's left
/// open rather than sealed. Leaving it open (instead of giving it its own
/// doorway hole, the way `portal_cube`'s walls do) avoids a wall panel
/// sitting exactly on the portal's own clip plane, which is unnecessary
/// here since — unlike `portal_cube` — nothing needs to recurse back out of
/// these terminal, single-destination rooms.
fn insert_room_shell(
    renderer: &mut Renderer,
    unit_mesh: helio::MeshId,
    wall_mat: helio::MaterialId,
    center: Vec3,
    half_size: f32,
    wall_t: f32,
    open_normal: Vec3,
) {
    let panels: [(Vec3, Vec3); 6] = [
        (Vec3::X, Vec3::new(wall_t, half_size, half_size)),
        (Vec3::NEG_X, Vec3::new(wall_t, half_size, half_size)),
        (Vec3::Y, Vec3::new(half_size, wall_t, half_size)),
        (Vec3::NEG_Y, Vec3::new(half_size, wall_t, half_size)),
        (Vec3::Z, Vec3::new(half_size, half_size, wall_t)),
        (Vec3::NEG_Z, Vec3::new(half_size, half_size, wall_t)),
    ];
    for (dir, half_extent) in panels {
        if dir.dot(open_normal) > 0.5 {
            continue;
        }
        insert_box_panel(renderer, unit_mesh, wall_mat, center + dir * half_size, half_extent);
    }
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
