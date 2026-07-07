//! Procedural backrooms map with VHS camcorder effect.
//!
//! Generates a random but reliably navigable map each run:
//! corridors, rooms, fluorescent lighting, all immersed in a
//! degraded VHS look (CA, grain, vignette, desaturated).
//!
//! Controls:
//!   WASD        — move
//!   Space/Shift — up/down
//!   R           — regenerate map
//!   Mouse drag  — look around

mod v3_demo_common;

use helio::{
    required_wgpu_features, required_wgpu_limits, Camera, DebugDrawState, GroupMask, HelioAction,
    HelioCommandBridge, LightId, MaterialId, MeshId, Movability, ObjectDescriptor, Renderer,
    RendererConfig, Scene,
};
use helio_default_graphs::build_default_graph_with_user_effects;
use helio_pass_postprocess::PostProcessPass;
use libhelio::{PostProcessSettings, PostProcessVolumeDescriptor};
use v3_demo_common::{box_mesh, make_material, point_light};

// User shader snippet injected into the post-process pipeline.
// Uses noise_tex, noise_samp, and pp_custom from the core bindings.
//
// v2 changes vs original, aimed at "found VHS footage" realism rather than
// a generic retro filter:
//   1. Lens identity: barrel distortion + vignette + edge chromatic aberration
//      (cheap camcorder wide-angle glass, not a flat digital frame)
//   2. CCD highlight bloom/smear: bright sources streak vertically, a classic
//      tell of consumer CCD sensors that's otherwise completely absent
//   3. Grain now samples noise_tex with a slow scroll instead of pure hash
//      noise -> spatially/temporally correlated grain instead of TV static
//   4. Head-switching noise bar near the bottom of frame (real decks show
//      this on every single frame - previously missing entirely)
//   5. Rare, severe tracking tears layered on top of the existing smooth
//      jitter, so damage reads as bursty/eventful instead of constant
// Everything from the original (YIQ separation, chroma phase drift, rolling
// scanline error, per-line jitter, crushed blacks/highlights, dot crawl,
// flicker) is kept - it was solid and just needed a physical frame around it.
const VHS_SHADER_SNIPPET: &str = include_str!("vhs_effects.wgsl");
use std::io::{self, BufRead};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use std::collections::HashSet;

// ── Map Generator ─────────────────────────────────────────────────────────────
//
// Simple grid-based generator: drunkard's walk carves corridors, rooms branch off.

const CELL: f32 = 4.0; // metres per grid cell
const H_CELL: f32 = CELL / 2.0;
const WALL_H: f32 = 3.2; // wall height
const GRID_W: usize = 32;
const GRID_H: usize = 24;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Cell {
    Wall,
    Corridor,
    Room,
}

struct BackroomsMap {
    grid: Vec<Vec<Cell>>,
    lights: Vec<(f32, f32)>, // world-space (x, z) for each light
}

fn generate_map() -> BackroomsMap {
    let mut grid = vec![vec![Cell::Wall; GRID_H]; GRID_W];

    // Start near centre
    let mut cx = GRID_W as i32 / 2;
    let mut cy = GRID_H as i32 / 2;
    grid[cx as usize][cy as usize] = Cell::Corridor;

    // Drunkard's walk — carve main corridor
    let steps = 140;
    let dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)];
    let mut rng = 12345_u64;

    let mut rng_u32 = || -> u32 {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng >> 32) as u32
    };

    for _ in 0..steps {
        let d = dirs[(rng_u32() as usize) & 3];
        let nx = cx + d.0;
        let ny = cy + d.1;
        if nx > 0 && nx < GRID_W as i32 - 1 && ny > 0 && ny < GRID_H as i32 - 1 {
            cx = nx;
            cy = ny;
            grid[cx as usize][cy as usize] = Cell::Corridor;
        }
    }

    // Place rooms branching off corridor cells
    for _ in 0..20 {
        // Pick a random corridor cell
        let rx = (rng_u32() as usize) % GRID_W;
        let ry = (rng_u32() as usize) % GRID_H;
        if grid[rx][ry] != Cell::Corridor {
            continue;
        }

        let rw = 2 + (rng_u32() % 3) as usize; // 2-4 cells wide
        let rh = 2 + (rng_u32() % 2) as usize; // 2-3 cells deep
        let rdir = dirs[(rng_u32() as usize) & 3];

        // Carve room in the chosen direction
        let mut ok = true;
        for dy in 0..rh {
            for dx in 0..rw {
                let gx = rx as i32 + rdir.0 * (1 + dx as i32);
                let gy = ry as i32 + rdir.1 * (1 + dy as i32);
                if gx < 1 || gx >= GRID_W as i32 - 1 || gy < 1 || gy >= GRID_H as i32 - 1 {
                    ok = false;
                } else if grid[gx as usize][gy as usize] != Cell::Wall {
                    ok = false;
                }
            }
        }
        if !ok {
            continue;
        }

        for dy in 0..rh {
            for dx in 0..rw {
                let gx = rx as i32 + rdir.0 * (1 + dx as i32);
                let gy = ry as i32 + rdir.1 * (1 + dy as i32);
                grid[gx as usize][gy as usize] = Cell::Room;
            }
        }
    }

    // Flood-fill connectivity: ensure all walkable cells reach the start
    let mut visited = vec![vec![false; GRID_H]; GRID_W];
    let start_x = GRID_W / 2;
    let start_y = GRID_H / 2;
    let mut stack = vec![(start_x, start_y)];
    visited[start_x][start_y] = true;
    while let Some((x, y)) = stack.pop() {
        for &(dx, dy) in &[(0, -1), (0, 1), (-1, 0), (1, 0)] {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx >= 0 && nx < GRID_W as i32 && ny >= 0 && ny < GRID_H as i32 {
                let (nx, ny) = (nx as usize, ny as usize);
                if !visited[nx][ny] && grid[nx][ny] != Cell::Wall {
                    visited[nx][ny] = true;
                    stack.push((nx, ny));
                }
            }
        }
    }

    // Remove unreachable walkable cells (turn them back to Wall)
    for x in 0..GRID_W {
        for y in 0..GRID_H {
            if grid[x][y] != Cell::Wall && !visited[x][y] {
                grid[x][y] = Cell::Wall;
            }
        }
    }

    // Place lights: one per walkable cell, slightly randomised
    let mut lights = Vec::new();
    for x in 0..GRID_W {
        for y in 0..GRID_H {
            if grid[x][y] != Cell::Wall {
                let wx = x as f32 * CELL - GRID_W as f32 * H_CELL + H_CELL;
                let wz = y as f32 * CELL - GRID_H as f32 * H_CELL + H_CELL;
                lights.push((wx, wz));
            }
        }
    }

    BackroomsMap { grid, lights }
}

// ── End Map Generator ─────────────────────────────────────────────────────────

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run");
}

// Ids we need to keep for regeneration
struct MapResources {
    walls: Vec<MeshId>,
    floors: Vec<MeshId>,
    ceilings: Vec<MeshId>,
    light_ids: Vec<LightId>,
}

struct App {
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    surface_format: wgpu::TextureFormat,
    renderer: Arc<Mutex<Renderer>>,
    action_rx: Receiver<HelioAction>,
    last_frame: std::time::Instant,

    map_resources: Option<MapResources>,

    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    start_time: std::time::Instant,
}

impl App {
    fn new() -> Self {
        Self { state: None }
    }

    fn place(
        scene: &mut Scene,
        mesh: MeshId,
        material: MaterialId,
        transform: glam::Mat4,
        radius: f32,
    ) {
        let _ = scene.insert_actor(helio::SceneActor::object(ObjectDescriptor {
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
            groups: GroupMask::NONE,
            movability: None,
            user_tag: 0,
        }));
    }

    fn regenerate_map(state: &mut AppState) {
        let map = generate_map();
        let mut renderer = state.renderer.lock().unwrap();
        let scene = renderer.scene_mut();

        // Remove previous map resources
        if let Some(res) = &state.map_resources {
            for &id in &res.walls {
                let _ = scene.remove_mesh(id);
            }
            for &id in &res.floors {
                let _ = scene.remove_mesh(id);
            }
            for &id in &res.ceilings {
                let _ = scene.remove_mesh(id);
            }
            for &id in &res.light_ids {
                let _ = scene.remove_light(id);
            }
        }

        let wall_mat = scene.insert_material(make_material(
            [0.82, 0.72, 0.52, 1.0],
            0.6,
            0.05,
            [0.0, 0.0, 0.0],
            0.0,
        ));
        let floor_mat = scene.insert_material(make_material(
            [0.45, 0.38, 0.28, 1.0],
            0.3,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));
        let ceiling_mat = scene.insert_material(make_material(
            [0.88, 0.88, 0.85, 1.0],
            0.7,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));
        let trim_mat = scene.insert_material(make_material(
            [0.35, 0.32, 0.28, 1.0],
            0.4,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));

        let mut walls = Vec::new();
        let mut floors = Vec::new();
        let mut ceilings = Vec::new();
        let mut light_ids = Vec::new();

        for x in 0..GRID_W {
            for y in 0..GRID_H {
                if map.grid[x][y] == Cell::Wall {
                    continue;
                }

                let wx = x as f32 * CELL - GRID_W as f32 * H_CELL + H_CELL;
                let wz = y as f32 * CELL - GRID_H as f32 * H_CELL + H_CELL;

                // Floor tile
                let f = scene
                    .insert_actor(helio::SceneActor::mesh(box_mesh(
                        [0.0, 0.0, 0.0],
                        [H_CELL, 0.05, H_CELL],
                    )))
                    .as_mesh()
                    .unwrap();
                Self::place(
                    scene,
                    f,
                    floor_mat,
                    glam::Mat4::from_translation(glam::Vec3::new(wx, 0.0, wz)),
                    H_CELL,
                );
                floors.push(f);

                // Ceiling tile
                let c = scene
                    .insert_actor(helio::SceneActor::mesh(box_mesh(
                        [0.0, 0.0, 0.0],
                        [H_CELL, 0.03, H_CELL],
                    )))
                    .as_mesh()
                    .unwrap();
                Self::place(
                    scene,
                    c,
                    ceiling_mat,
                    glam::Mat4::from_translation(glam::Vec3::new(wx, WALL_H, wz)),
                    H_CELL,
                );
                ceilings.push(c);

                // Walls on edges adjacent to Wall cells
                let neighbors = [
                    (0, -1, 0.0, -H_CELL, 0.0),                               // south
                    (0, 1, 0.0, H_CELL, std::f32::consts::PI),                // north
                    (-1, 0, -H_CELL, 0.0, std::f32::consts::FRAC_PI_2 * 3.0), // west
                    (1, 0, H_CELL, 0.0, std::f32::consts::FRAC_PI_2),         // east
                ];
                for &(dx, dy, ox, oz, rot_y) in &neighbors {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let is_wall = nx < 0
                        || nx >= GRID_W as i32
                        || ny < 0
                        || ny >= GRID_H as i32
                        || map.grid[nx as usize][ny as usize] == Cell::Wall;
                    if !is_wall {
                        continue;
                    }

                    // Wall
                    let w = scene
                        .insert_actor(helio::SceneActor::mesh(box_mesh(
                            [0.0, 0.0, 0.0],
                            [0.1, WALL_H / 2.0, CELL / 2.0],
                        )))
                        .as_mesh()
                        .unwrap();
                    let t = glam::Mat4::from_translation(glam::Vec3::new(
                        wx + ox,
                        WALL_H / 2.0,
                        wz + oz,
                    )) * glam::Mat4::from_rotation_y(rot_y);
                    Self::place(scene, w, wall_mat, t, H_CELL);
                    walls.push(w);

                    // Baseboard trim
                    let t2 = scene
                        .insert_actor(helio::SceneActor::mesh(box_mesh(
                            [0.0, 0.0, 0.0],
                            [0.12, 0.05, CELL / 2.0],
                        )))
                        .as_mesh()
                        .unwrap();
                    let tt = glam::Mat4::from_translation(glam::Vec3::new(wx + ox, 0.05, wz + oz))
                        * glam::Mat4::from_rotation_y(rot_y);
                    Self::place(scene, t2, trim_mat, tt, H_CELL);
                    walls.push(t2);
                }
            }
        }

        // Fluorescent lights
        let light_colors = [
            [0.95, 0.92, 0.85],
            [0.92, 0.93, 0.88],
            [0.96, 0.91, 0.82],
            [0.90, 0.94, 0.86],
        ];
        let mut rng = 54321_u64;
        let mut rng_u32 = || -> u32 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 32) as u32
        };
        for &(lx, lz) in &map.lights {
            let ci = (rng_u32() as usize) % light_colors.len();
            let flicker_offset = (rng_u32() as f32 / u32::MAX as f32) * 100.0;
            let _ = scene.insert_actor(helio::SceneActor::light_with_movability(
                point_light([lx, WALL_H - 0.2, lz], light_colors[ci], 3.5, 8.0),
                Some(helio::Movability::Movable),
            ));
            // Also add a dimmer fill light slightly lower
            light_ids.push(
                scene
                    .insert_actor(helio::SceneActor::light_with_movability(
                        point_light([lx, WALL_H - 1.5, lz], [0.95, 0.92, 0.85], 1.2, 4.0),
                        Some(helio::Movability::Movable),
                    ))
                    .as_light()
                    .unwrap(),
            );
        }

        state.map_resources = Some(MapResources {
            walls,
            floors,
            ceilings,
            light_ids,
        });
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
                        .with_title("Helio – VHS Backrooms")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .expect("window"),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_features: required_wgpu_features(adapter.features()),
            required_limits: required_wgpu_limits(adapter.limits()),
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
            },
        );

        let config = RendererConfig::new(size.width, size.height, format)
            .with_shadow_quality(helio::ShadowQuality::Ultra)
            .with_render_scale(1.0);
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
        let graph = build_default_graph_with_user_effects(
            &device,
            &queue,
            &scene,
            config,
            debug_state.clone(),
            &debug_camera_buf,
            &cull_stats_buf,
            None,
            VHS_SHADER_SNIPPET,
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

        // ── VHS camcorder post-process volume ─────────────────────────────────
        renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::post_process_volume(
                PostProcessVolumeDescriptor {
                    bounds_min: [-1000.0, -1000.0, -1000.0],
                    bounds_max: [1000.0, 1000.0, 1000.0],
                    blend_radius: 0.0,
                    unbound: true,
                    priority: 100.0,
                    blend_weight: 1.0,
                    settings: PostProcessSettings {
                        // All effects are handled by the user_effects WGSL snippet.
                        // Keep the volume at defaults so the built-in chain is a
                        // no-op — the VHS shader owns the entire post-process look.
                        ..PostProcessSettings::default()
                    },
                },
            ));

        renderer.set_ambient([0.75, 0.7, 0.6], 0.04);
        renderer.set_clear_color([0.0, 0.0, 0.0, 1.0]);

        let renderer = Arc::new(Mutex::new(renderer));
        let (bridge, action_rx) = HelioCommandBridge::new();
        let command_bridge = Arc::new(bridge);

        {
            let bridge = command_bridge.clone();
            std::thread::spawn(move || {
                let stdin = io::stdin();
                for line in stdin.lock().lines() {
                    match line {
                        Ok(cmd) if !cmd.trim().is_empty() => match bridge.run(&cmd) {
                            Ok(()) => println!("OK: {}", cmd),
                            Err(e) => println!("ERR: {} -> {}", cmd, e),
                        },
                        _ => {}
                    }
                }
            });
        }

        self.state = Some(AppState {
            window,
            surface,
            device,
            surface_format: format,
            renderer,
            action_rx,
            last_frame: std::time::Instant::now(),
            map_resources: None,
            cam_pos: glam::Vec3::new(0.0, 1.6, 0.0),
            cam_yaw: 0.0,
            cam_pitch: 0.0,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            start_time: std::time::Instant::now(),
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
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::KeyR),
                        ..
                    },
                ..
            } => {
                App::regenerate_map(state);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(code),
                        ..
                    },
                ..
            } => {
                state.keys.insert(code);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Released,
                        physical_key: PhysicalKey::Code(code),
                        ..
                    },
                ..
            } => {
                state.keys.remove(&code);
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if !state.cursor_grabbed {
                    state.cursor_grabbed = true;
                    let _ = state.window.set_cursor_grab(CursorGrabMode::Locked);
                    state.window.set_cursor_visible(false);
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
                    },
                );
                if let Ok(mut renderer) = state.renderer.lock() {
                    renderer.set_render_size(s.width, s.height);
                }
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
        // Generate map on first render (after renderer is fully set up)
        if self.map_resources.is_none() {
            App::regenerate_map(self);
        }

        const SPEED: f32 = 3.0;
        const SENS: f32 = 0.002;

        self.cam_yaw += self.mouse_delta.0 * SENS;
        self.cam_pitch = (self.cam_pitch - self.mouse_delta.1 * SENS).clamp(-1.2, 1.2);
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

        let time = self.start_time.elapsed().as_secs_f32();
        let bob_amt = 0.015;
        let bob_speed = 3.5;
        let bob = (time * bob_speed).sin() * bob_amt;
        let bob_sway = (time * bob_speed * 0.5).cos() * bob_amt * 0.5;

        let cam_pos = self.cam_pos + glam::Vec3::new(bob_sway, bob, 0.0);

        let size = self.window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;

        let camera = Camera::perspective_look_at(
            cam_pos,
            cam_pos + forward,
            glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            aspect,
            0.1,
            200.0,
        );

        let mut renderer = self.renderer.lock().unwrap();
        while let Ok(action) = self.action_rx.try_recv() {
            match action {
                HelioAction::SetDebugMode(mode) => renderer.set_debug_mode(mode),
                HelioAction::SetEditorMode(enabled) => renderer.set_editor_mode(enabled),
                HelioAction::DebugClear => renderer.debug_clear(),
            }
        }

        // Write VHS parameters to post-process custom params buffer
        let time = self.start_time.elapsed().as_secs_f32();
        let vhs_params: [[f32; 4]; 2] = [
            [
                0.0,  // unused
                0.12, // tape jitter — ~0.6 px max horizontal displacement
                8.0,  // jitter frequency
                0.2,
            ], // flicker intensity
            [
                0.4,  // noise amount
                time, // animation time
                0.0, 0.0,
            ],
        ];
        if let Some(pass) = renderer.find_pass_mut::<helio_pass_postprocess::PostProcessPass>() {
            pass.set_custom_params(&vhs_params);
        }

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => {
                log::warn!("Surface: {:?}", e);
                return;
            }
        };
        let view = output.texture.create_view(&Default::default());

        if let Err(e) = renderer.render(&camera, &view) {
            log::error!("Render: {:?}", e);
        }
        output.present();
    }
}

