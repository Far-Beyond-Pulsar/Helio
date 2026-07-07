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
    required_wgpu_features, required_wgpu_limits, Camera, DebugDrawState, HelioAction,
    HelioCommandBridge, GroupMask, LightId, MaterialId, MeshId, Movability, ObjectDescriptor,
    Renderer, RendererConfig, Scene,
};
use helio_default_graphs::build_default_graph_with_user_effects;
use helio_pass_postprocess::PostProcessPass;
use libhelio::{PostProcessSettings, PostProcessVolumeDescriptor, TonemapOperator};
use v3_demo_common::{box_mesh, make_material, point_light};

// User shader snippet injected into the post-process pipeline.
// Uses noise_tex, noise_samp, and pp_custom from the core bindings.
const VHS_SHADER_SNIPPET: &str = "
fn hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453123);
}

fn yiq2rgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c, vec3<f32>(1.0,  0.956,  0.621)),
        dot(c, vec3<f32>(1.0, -0.272, -0.647)),
        dot(c, vec3<f32>(1.0, -1.106,  1.703)),
    );
}

fn rgb2yiq(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c, vec3<f32>(0.299, 0.587, 0.114)),
        dot(c, vec3<f32>(0.596, -0.274, -0.322)),
        dot(c, vec3<f32>(0.211, -0.523, 0.312)),
    );
}

fn blur_ring(uv: vec2<f32>, radius: f32, n: u32) -> vec3<f32> {
    var acc = vec3<f32>(0.0);
    let inv = 1.0 / f32(n);
    for (var i = 0u; i < n; i++) {
        let a = 6.2832 * f32(i) * inv;
        acc += textureSampleLevel(hdr_input, linear_samp, uv + vec2<f32>(cos(a), sin(a)) * radius, 0.0).rgb;
    }
    return acc * inv;
}

fn user_effects(color: vec3<f32>, uv: vec2<f32>, dims: vec2<f32>) -> vec3<f32> {
    let tape_jitter  = pp_custom[0].y;
    let jitter_freq  = pp_custom[0].z;
    let flicker_amt  = pp_custom[0].w;
    let noise_amt    = pp_custom[1].x;
    let time         = pp_custom[1].y;

    let px = 1.0 / dims;
    let frame = floor(time * 60.0);

    let line = uv.y * dims.y;

    // ── Vertical bounce (VCR servo instability) ───────────────────────────
    // The entire frame slowly bobs up/down from capstan servo fluctuations.
    let v_bounce = sin(time * 1.73) * 0.5 + sin(time * 3.11) * 0.25;
    let vu = uv + vec2<f32>(0.0, v_bounce * px.y);

    // ── Rolling scanline offset error ─────────────────────────────────────
    let roll_pos = fract(time * 0.025 + 0.3);
    let roll_width = 0.06;
    let roll_dist = abs(vu.y - roll_pos);
    let roll_weight = 1.0 - smoothstep(0.0, roll_width, roll_dist);
    let roll_shift = sin(line * 50.0 + time * 2.0) * roll_weight * 2.5;

    // ── Per-scanline tape jitter ──────────────────────────────────────────
    let jit = (sin(line * jitter_freq + time * 3.7) * 5.0
             + sin(line * 17.0 + time * 5.3) * 2.0) * tape_jitter;
    let ju = vu + vec2<f32>((jit + roll_shift) * px.x, 0.0);

    // ── YIQ separation with per-channel blur ─────────────────────────────
    let yuv = blur_ring(ju, 0.5 * px.x, 5u);
    let y = rgb2yiq(yuv).x;

    let i_uv = ju + vec2<f32>(0.6 * px.x, 0.0);
    let i_base = rgb2yiq(blur_ring(i_uv, 3.0 * px.x, 9u)).y;

    let q_uv = ju + vec2<f32>(1.2 * px.x, 0.0);
    let q_base = rgb2yiq(blur_ring(q_uv, 1.5 * px.x, 9u)).z;

    // ── Chroma phase drift (VHS color decoder instability) ────────────────
    // The chroma subcarrier PLL slowly drifts, rotating the hue.
    let phase = sin(time * 0.37) * 0.08 + sin(time * 0.73) * 0.04;
    let cp = cos(phase);
    let sp = sin(phase);
    let i = i_base * cp - q_base * sp;
    let q = i_base * sp + q_base * cp;

    var result = yiq2rgb(vec3<f32>(y, i, q));

    // ── Rolling glitch noise bands ────────────────────────────────────────
    // Thin noise bars that roll vertically like tracking comb-filter errors,
    // with smooth intensity fade instead of abrupt random pops.
    let glitch_roll = fract(time * 0.018 + 0.5);
    for (var g = 0u; g < 4u; g++) {
        let gp = fract(glitch_roll + f32(g) * 0.25 + sin(time * 0.1 + f32(g)) * 0.02);
        let gd = abs(vu.y - gp);
        let gw = 1.0 - smoothstep(0.0, 0.025, gd);
        if gw > 0.005 {
            let gn = hash21(vec2<f32>(line + f32(g) * 100.0, frame));
            result = mix(result, vec3<f32>(gn * 0.4 + 0.1), gw * 0.7);
        }
    }

    // ── VHS lighting: crushed blacks, clipped highlights ─────────────────
    result = pow(max(result, vec3<f32>(0.0)), vec3<f32>(1.4));
    result = 1.0 - pow(max(1.0 - result, vec3<f32>(0.0)), vec3<f32>(1.6));
    result = mix(result, result * result * result, 0.15);

    // ── Scanlines ─────────────────────────────────────────────────────────
    let scan = sin(uv.y * dims.y * 3.14159);
    result *= 1.0 - 0.06 * (1.0 - scan * scan);

    // ── Non-sliding hash noise ────────────────────────────────────────────
    let px_id = floor(uv * dims);
    let seed = frame + hash21(px_id) * 1000.0;
    let r0 = hash21(vec2<f32>(seed, seed * 0.5));
    let r1 = hash21(vec2<f32>(seed + 1.0, seed * 0.3 + 2.0));

    let luma = dot(result, vec3<f32>(0.299, 0.587, 0.114));
    let noise_strength = (0.015 + 0.03 * (1.0 - luma)) * noise_amt;
    let grain = (r0 * 2.0 - 1.0) * noise_strength;
    result += grain;

    // ── Chroma noise ──────────────────────────────────────────────────────
    let cn = (r1 * 2.0 - 1.0) * 0.03 * (1.0 - luma) * noise_amt;
    result += vec3<f32>(cn * 0.3, -cn * 0.5, cn * 0.7);

    // ── Dot crawl on sharp edges ──────────────────────────────────────────
    let edge = length(fwidth(result));
    let crawl = sin(uv.x * dims.x * 0.5 + time * 50.0) * edge * 0.06;
    result += vec3<f32>(crawl * 0.4, -crawl * 0.25, crawl * 0.6);

    // ── Flicker ───────────────────────────────────────────────────────────
    let fl = hash21(vec2<f32>(frame * 0.01, 0.5));
    result *= 1.0 - 0.025 * flicker_amt * (fl * 2.0 - 1.0);

    return clamp(result, vec3<f32>(0.0), vec3<f32>(1.0));
}
";

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

const CELL: f32 = 4.0;         // metres per grid cell
const H_CELL: f32 = CELL / 2.0;
const WALL_H: f32 = 3.2;       // wall height
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
    lights: Vec<(f32, f32)>,   // world-space (x, z) for each light
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
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
        if grid[rx][ry] != Cell::Corridor { continue; }

        let rw = 2 + (rng_u32() % 3) as usize;  // 2-4 cells wide
        let rh = 2 + (rng_u32() % 2) as usize;  // 2-3 cells deep
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
        if !ok { continue; }

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

    fn place(scene: &mut Scene, mesh: MeshId, material: MaterialId, transform: glam::Mat4, radius: f32) {
        let _ = scene.insert_actor(helio::SceneActor::object(ObjectDescriptor {
            mesh,
            material,
            transform,
            bounds: [transform.w_axis.x, transform.w_axis.y, transform.w_axis.z, radius],
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
            for &id in &res.walls { let _ = scene.remove_mesh(id); }
            for &id in &res.floors { let _ = scene.remove_mesh(id); }
            for &id in &res.ceilings { let _ = scene.remove_mesh(id); }
            for &id in &res.light_ids { let _ = scene.remove_light(id); }
        }

        let wall_mat = scene.insert_material(make_material(
            [0.82, 0.72, 0.52, 1.0], 0.6, 0.05, [0.0, 0.0, 0.0], 0.0,
        ));
        let floor_mat = scene.insert_material(make_material(
            [0.45, 0.38, 0.28, 1.0], 0.3, 0.0, [0.0, 0.0, 0.0], 0.0,
        ));
        let ceiling_mat = scene.insert_material(make_material(
            [0.88, 0.88, 0.85, 1.0], 0.7, 0.0, [0.0, 0.0, 0.0], 0.0,
        ));
        let trim_mat = scene.insert_material(make_material(
            [0.35, 0.32, 0.28, 1.0], 0.4, 0.0, [0.0, 0.0, 0.0], 0.0,
        ));

        let mut walls = Vec::new();
        let mut floors = Vec::new();
        let mut ceilings = Vec::new();
        let mut light_ids = Vec::new();

        for x in 0..GRID_W {
            for y in 0..GRID_H {
                if map.grid[x][y] == Cell::Wall { continue; }

                let wx = x as f32 * CELL - GRID_W as f32 * H_CELL + H_CELL;
                let wz = y as f32 * CELL - GRID_H as f32 * H_CELL + H_CELL;

                // Floor tile
                let f = scene.insert_actor(helio::SceneActor::mesh(box_mesh(
                    [0.0, 0.0, 0.0], [H_CELL, 0.05, H_CELL],
                ))).as_mesh().unwrap();
                Self::place(scene, f, floor_mat,
                    glam::Mat4::from_translation(glam::Vec3::new(wx, 0.0, wz)), H_CELL);
                floors.push(f);

                // Ceiling tile
                let c = scene.insert_actor(helio::SceneActor::mesh(box_mesh(
                    [0.0, 0.0, 0.0], [H_CELL, 0.03, H_CELL],
                ))).as_mesh().unwrap();
                Self::place(scene, c, ceiling_mat,
                    glam::Mat4::from_translation(glam::Vec3::new(wx, WALL_H, wz)), H_CELL);
                ceilings.push(c);

                // Walls on edges adjacent to Wall cells
                let neighbors = [
                    (0, -1, 0.0, -H_CELL, 0.0),   // south
                    (0, 1, 0.0, H_CELL, std::f32::consts::PI), // north
                    (-1, 0, -H_CELL, 0.0, std::f32::consts::FRAC_PI_2 * 3.0), // west
                    (1, 0, H_CELL, 0.0, std::f32::consts::FRAC_PI_2), // east
                ];
                for &(dx, dy, ox, oz, rot_y) in &neighbors {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let is_wall = nx < 0 || nx >= GRID_W as i32 || ny < 0 || ny >= GRID_H as i32
                        || map.grid[nx as usize][ny as usize] == Cell::Wall;
                    if !is_wall { continue; }

                    // Wall
                    let w = scene.insert_actor(helio::SceneActor::mesh(box_mesh(
                        [0.0, 0.0, 0.0], [0.1, WALL_H / 2.0, CELL / 2.0],
                    ))).as_mesh().unwrap();
                    let t = glam::Mat4::from_translation(glam::Vec3::new(wx + ox, WALL_H / 2.0, wz + oz))
                        * glam::Mat4::from_rotation_y(rot_y);
                    Self::place(scene, w, wall_mat, t, H_CELL);
                    walls.push(w);

                    // Baseboard trim
                    let t2 = scene.insert_actor(helio::SceneActor::mesh(box_mesh(
                        [0.0, 0.0, 0.0], [0.12, 0.05, CELL / 2.0],
                    ))).as_mesh().unwrap();
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
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
                scene.insert_actor(helio::SceneActor::light_with_movability(
                    point_light([lx, WALL_H - 1.5, lz], [0.95, 0.92, 0.85], 1.2, 4.0),
                    Some(helio::Movability::Movable),
                )).as_light().unwrap(),
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
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
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
        renderer.scene_mut().insert_actor(helio::SceneActor::post_process_volume(
            PostProcessVolumeDescriptor {
                bounds_min: [-1000.0, -1000.0, -1000.0],
                bounds_max: [1000.0, 1000.0, 1000.0],
                blend_radius: 0.0,
                unbound: true,
                priority: 100.0,
                blend_weight: 1.0,
                settings: PostProcessSettings {
                    ca_intensity: 0.015,
                    ca_start_offset: 0.1,
                    ca_enabled: true,
                    grain_intensity: 0.25,
                    grain_response: 1.5,
                    grain_size: 1.5,
                    grain_enabled: true,
                    vignette_intensity: 0.7,
                    vignette_smoothness: 2.5,
                    vignette_roundness: 0.8,
                    vignette_color: [0.0, 0.0, 0.0],
                    vignette_enabled: true,
                    color_saturation: [0.6, 0.55, 0.5],
                    color_contrast: [1.1, 1.08, 1.05],
                    color_gamma: [0.95, 0.95, 0.95],
                    color_gain: [0.95, 0.9, 0.85],
                    color_offset: [0.0, 0.01, 0.0],
                    bloom_intensity: 0.8,
                    bloom_threshold: 0.8,
                    bloom_knee: 0.3,
                    bloom_enabled: true,
                    bloom_tint: [1.0, 0.95, 0.85],
                    tonemap_operator: TonemapOperator::Aces,
                    tonemap_exposure: 0.85,
                    tonemap_white_point: 1.0,
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
                        Ok(cmd) if !cmd.trim().is_empty() => {
                            match bridge.run(&cmd) {
                                Ok(()) => println!("OK: {}", cmd),
                                Err(e) => println!("ERR: {} -> {}", cmd, e),
                            }
                        }
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
            [0.0,    // unused
             0.12,   // tape jitter — ~0.6 px max horizontal displacement
             8.0,    // jitter frequency
             0.2],   // flicker intensity
            [0.4,    // noise amount
             time,   // animation time
             0.0, 0.0],
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
