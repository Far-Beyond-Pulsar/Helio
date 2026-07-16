//! Indoor cathedral decal demo – projects decals onto the cathedral geometry
//!
//! Demonstrates the decal rendering system: bullet holes, cracks, blood
//! splatters, wear marks, soot, and glowing runes — all projected onto
//! the existing G-buffer without modifying geometry.
//!
//! Decals placed:
//!   • Worn path on the nave floor (multiply blend)
//!   • Fading blood splatters near the altar (translucent)
//!   • Crack on a stone column (normal-only perturbation)
//!   • Graffiti / sticker on the left wall (alpha blend)
//!   • Bullet holes on the right wall near entrance (alpha blend)
//!   • Soot marks on the ceiling under each chandelier (translucent)
//!   • A glowing rune on the altar (emissive, additive blend)
//!   • Age-fading decal near the entrance that decays over time
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   F1          — cycle debug modes
//!   F2          — toggle performance overlay
//!   F3          — toggle debug overlay
//!   F5          — spawn a fresh batch of bullet-hole decals at camera target
//!   Mouse drag  — look around (click to grab cursor)
//!   Escape      — release cursor / exit

mod v3_demo_common;

use helio::{
    required_wgpu_features, required_wgpu_limits, BakeConfig, Camera, DebugDrawState, DecalId,
    HelioAction, HelioCommandBridge, LightId, MeshId, Movability, Renderer, RendererConfig, Scene,
};
use libhelio::{DecalBlendMode, DecalType, GpuDecal};
use helio_pass_perf_overlay::PerfOverlayMode;
use helio_default_graphs::build_default_graph;
use v3_demo_common::{box_mesh, make_material, plane_mesh, point_light};

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

/// Build a world-to-decal-space transform for a decal at `position`
/// facing `normal`, with the given half-extents.
fn decal_transform(position: [f32; 3], normal: [f32; 3], half_extents: [f32; 3]) -> [f32; 16] {
    let p = glam::Vec3::from_array(position);
    let n = glam::Vec3::from_array(normal).normalize();
    let rot = glam::Quat::from_rotation_arc(glam::Vec3::Z, n);
    // world-to-decal: translate to origin, rotate to align Z with normal, scale to [-1,1]³
    let m = glam::Mat4::from_scale(glam::Vec3::new(
        1.0 / half_extents[0],
        1.0 / half_extents[1],
        1.0 / half_extents[2],
    )) * glam::Mat4::from_quat(rot)
        * glam::Mat4::from_translation(-p);
    m.to_cols_array()
}

fn make_decal(
    position: [f32; 3],
    normal: [f32; 3],
    half_extents: [f32; 3],
    color: [f32; 4],
    blend_mode: DecalBlendMode,
    decal_type: DecalType,
    normal_adapt: bool,
    fade_time: f32,
    fade_start_delay: f32,
) -> GpuDecal {
    GpuDecal {
        transform: decal_transform(position, normal, half_extents),
        color,
        albedo_texture_index: u32::MAX,
        normal_texture_index: u32::MAX,
        roughness_texture_index: u32::MAX,
        metalness_texture_index: u32::MAX,
        blend_mode: blend_mode as u32,
        decal_type: decal_type as u32,
        fade_time,
        fade_start_delay,
        age: 0.0,
        normal_adapt: if normal_adapt { 1 } else { 0 },
        _pad0: 0.0,
        _pad1: 0.0,
    }
}

// ── Scene data ────────────────────────────────────────────────────────────────

// Column positions along the nave (Z axis), symmetric at x = ±5.5
const COLUMN_Z: &[f32] = &[-22.0, -14.0, -6.0, 2.0, 10.0, 18.0];

// Stained glass window lights: (x_wall_side, y, z, r, g, b)
// Positive x = right-side windows, negative = left-side; placed just inside the wall
const GLASS_LIGHTS: &[(f32, f32, f32, f32, f32, f32)] = &[
    // Left wall (x ≈ -10.5), windows between columns
    (-10.3, 9.0, -18.0, 0.8, 0.2, 1.0), // violet
    (-10.3, 9.0, -6.0, 0.2, 0.7, 1.0),  // sky blue
    (-10.3, 9.0, 6.0, 0.2, 1.0, 0.4),   // emerald
    (-10.3, 9.0, 18.0, 1.0, 0.7, 0.1),  // gold
    // Right wall (x ≈ +10.5)
    (10.3, 9.0, -18.0, 1.0, 0.2, 0.3), // ruby
    (10.3, 9.0, -6.0, 1.0, 0.5, 0.1),  // amber
    (10.3, 9.0, 6.0, 0.1, 0.8, 0.9),   // teal
    (10.3, 9.0, 18.0, 0.9, 0.1, 0.7),  // magenta
    // Rose window above entrance (back wall, z ≈ +28)
    (0.0, 13.0, 27.0, 1.0, 0.75, 0.3), // warm gold
];

// Chandelier positions (x=0, hanging from y≈19.5, at z intervals)
const CHANDELIER_Z: &[f32] = &[-16.0, 0.0, 16.0];

// Candle cluster positions near the altar (z ≈ -24)
const CANDLES: &[(f32, f32, f32)] = &[
    (-3.0, 1.6, -23.5),
    (-1.5, 1.6, -23.0),
    (0.0, 1.6, -23.5),
    (1.5, 1.6, -23.0),
    (3.0, 1.6, -23.5),
];

// Pew rows: 6 per side, spaced 2.4 m apart starting at z = -20
const PEW_Z_START: f32 = -20.0;
const PEW_Z_STEP: f32 = 3.2;
const PEW_COUNT: usize = 6;

// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run");
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

    // Major structural surfaces
    _floor: MeshId,
    _nave_ceiling: MeshId,
    _aisle_ceil_l: MeshId,
    _aisle_ceil_r: MeshId,
    _wall_left_outer: MeshId,
    _wall_right_outer: MeshId,
    _wall_front: MeshId,
    _wall_back: MeshId,
    // Colonnade arches (inner walls between nave and aisles, with gaps left for columns)
    _colonnade_l: Vec<MeshId>, // wall segments between columns
    _colonnade_r: Vec<MeshId>,
    // Columns
    _columns: Vec<MeshId>,
    // Altar
    _altar_plinth: MeshId,
    _altar_step: MeshId,
    _cross_vert: MeshId,
    _cross_horiz: MeshId,
    // Pews
    _pews_left: Vec<MeshId>,
    _pews_right: Vec<MeshId>,
    // Chandelier bodies (chain + ring)
    _chandelier_chains: Vec<MeshId>,
    _chandelier_rings: Vec<MeshId>,

    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    // Debug
    debug_mode: u32,
    perf_overlay_mode: PerfOverlayMode,
    debug_overlay_enabled: bool,

    // Scene state
    chandelier_light_ids: Vec<LightId>,
    candle_light_ids: Vec<LightId>,
    start_time: std::time::Instant,

    // Decals
    decal_ids: Vec<DecalId>,
    fading_decal_id: Option<DecalId>,
    bullet_hole_ids: Vec<DecalId>,
    bullet_hole_spawn_timer: f32,
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
                        .with_title("Helio – Indoor Cathedral")
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
            apply_limit_buckets: true,
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
                color_space: wgpu::SurfaceColorSpace::Auto,
            },
        );

        let config = RendererConfig::new(size.width, size.height, format)
                .with_shadow_quality(helio::ShadowQuality::Ultra);
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
        renderer.set_editor_mode(true);

        let mat = renderer.scene_mut().insert_material(make_material(
            [0.75, 0.72, 0.68, 1.0],
            0.85,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));

        renderer.scene_mut().insert_actor(helio::SceneActor::Sky(
            helio::SkyActor::indoor([0.05, 0.05, 0.1]).with_clouds(helio::VolumetricClouds {
                coverage: 0.7,
                density: 0.8,
                base: 1200.0,
                top: 1800.0,
                wind_x: 0.8,
                wind_z: 0.2,
                speed: 1.3,
                skylight_intensity: 0.25,
            }),
        ));

        // Nave + aisles: total width = 22m (x: -11..+11), length = 60m (z: -28..+28), height = 21m
        // Expand floor to cover full cathedral footprint. 32m radius = 64m square.
        let _floor =            renderer.scene_mut().insert_actor(helio::SceneActor::mesh(plane_mesh([0.0, 0.0, 0.0], 32.0))).as_mesh().unwrap();
        let _wall_back =        renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [11.0, 10.5, 0.25]))).as_mesh().unwrap();
        let _wall_front =       renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [11.0, 10.5, 0.25]))).as_mesh().unwrap();
        let _aisle_ceil_l =     renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [2.5, 0.15, 28.0]))).as_mesh().unwrap();
        let _nave_ceiling =     renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [6.0, 0.18, 28.0]))).as_mesh().unwrap();
        let _aisle_ceil_r =     renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [2.5, 0.15, 28.0]))).as_mesh().unwrap();
        let _wall_left_outer =  renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [0.25, 7.0, 28.0]))).as_mesh().unwrap();
        let _wall_right_outer = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [0.25, 7.0, 28.0]))).as_mesh().unwrap();
        let _ =
            v3_demo_common::insert_object(&mut renderer, _floor, mat, glam::Mat4::IDENTITY, 11.0);
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            _nave_ceiling,
            mat,
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 21.0, 0.0)),
            28.0,
        );
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            _aisle_ceil_l,
            mat,
            glam::Mat4::from_translation(glam::Vec3::new(-8.5, 11.0, 0.0)),
            28.0,
        );
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            _aisle_ceil_r,
            mat,
            glam::Mat4::from_translation(glam::Vec3::new(8.5, 11.0, 0.0)),
            28.0,
        );
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            _wall_left_outer,
            mat,
            glam::Mat4::from_translation(glam::Vec3::new(-11.0, 7.0, 0.0)),
            28.0,
        );
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            _wall_right_outer,
            mat,
            glam::Mat4::from_translation(glam::Vec3::new(11.0, 7.0, 0.0)),
            28.0,
        );
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            _wall_front,
            mat,
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 10.5, 28.0)),
            11.0,
        );
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            _wall_back,
            mat,
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 10.5, -28.0)),
            11.0,
        );

        // Colonnade: short wall segments between columns (between column z-positions)
        // 7 segments per side: before first col, between each pair, after last col
        let col_z_all: Vec<f32> = {
            let mut v = vec![-28.0_f32]; // south wall
            v.extend_from_slice(COLUMN_Z);
            v.push(28.0); // north wall
            v
        };
        let _colonnade_l: Vec<MeshId> = col_z_all
            .windows(2)
            .map(|w| {
                let mid_z = (w[0] + w[1]) * 0.5;
                let half_len = (w[1] - w[0]) * 0.5 - 0.9; // gap for column
                let id = renderer.scene_mut()
                    .insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [0.25, 5.5, half_len.max(0.1)])))
                    .as_mesh()
                    .unwrap();
                let _ = v3_demo_common::insert_object(
                    &mut renderer,
                    id,
                    mat,
                    glam::Mat4::from_translation(glam::Vec3::new(-5.5, 5.5, mid_z)),
                    5.5,
                );
                id
            })
            .collect();
        let _colonnade_r: Vec<MeshId> = col_z_all
            .windows(2)
            .map(|w| {
                let mid_z = (w[0] + w[1]) * 0.5;
                let half_len = (w[1] - w[0]) * 0.5 - 0.9;
                let id = renderer.scene_mut()
                    .insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [0.25, 5.5, half_len.max(0.1)])))
                    .as_mesh()
                    .unwrap();
                let _ = v3_demo_common::insert_object(
                    &mut renderer,
                    id,
                    mat,
                    glam::Mat4::from_translation(glam::Vec3::new(5.5, 5.5, mid_z)),
                    5.5,
                );
                id
            })
            .collect();

        // Columns: 0.65 m square, 20 m tall, at x = ±5.5
        let _columns: Vec<MeshId> = COLUMN_Z
            .iter()
            .flat_map(|&z| {
                let l = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [0.65, 10.0, 0.65]))).as_mesh().unwrap();
                let _ = v3_demo_common::insert_object(
                    &mut renderer,
                    l,
                    mat,
                    glam::Mat4::from_translation(glam::Vec3::new(-5.5, 10.0, z)),
                    10.0,
                );
                let r = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [0.65, 10.0, 0.65]))).as_mesh().unwrap();
                let _ = v3_demo_common::insert_object(
                    &mut renderer,
                    r,
                    mat,
                    glam::Mat4::from_translation(glam::Vec3::new(5.5, 10.0, z)),
                    10.0,
                );
                [l, r]
            })
            .collect();

        // Altar: at far end (z = -26)
        let _altar_step = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [5.5, 0.20, 3.0]))).as_mesh().unwrap();
        let _altar_plinth = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [3.0, 0.45, 1.5]))).as_mesh().unwrap();
        let _cross_vert = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [0.18, 2.2, 0.18]))).as_mesh().unwrap();
        let _cross_horiz = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [1.0, 0.18, 0.18]))).as_mesh().unwrap();
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            _altar_step,
            mat,
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.2, -24.5)),
            5.5,
        );
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            _altar_plinth,
            mat,
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.65, -25.5)),
            3.0,
        );
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            _cross_vert,
            mat,
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 3.2, -25.8)),
            2.2,
        );
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            _cross_horiz,
            mat,
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 4.5, -25.8)),
            1.0,
        );

        // Pews: long narrow rect3d per row, 6 rows each side
        let _pews_left: Vec<MeshId> = (0..PEW_COUNT)
            .map(|i| {
                let z = PEW_Z_START + i as f32 * PEW_Z_STEP;
                let id = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [1.5, 0.45, 0.5]))).as_mesh().unwrap();
                let _ = v3_demo_common::insert_object(
                    &mut renderer,
                    id,
                    mat,
                    glam::Mat4::from_translation(glam::Vec3::new(-3.2, 0.45, z)),
                    1.5,
                );
                id
            })
            .collect();
        let _pews_right: Vec<MeshId> = (0..PEW_COUNT)
            .map(|i| {
                let z = PEW_Z_START + i as f32 * PEW_Z_STEP;
                let id = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [1.5, 0.45, 0.5]))).as_mesh().unwrap();
                let _ = v3_demo_common::insert_object(
                    &mut renderer,
                    id,
                    mat,
                    glam::Mat4::from_translation(glam::Vec3::new(3.2, 0.45, z)),
                    1.5,
                );
                id
            })
            .collect();

        // Chandeliers: vertical chain + horizontal ring at each Z
        let chandelier_mat = renderer.scene_mut().insert_material(make_material(
            [0.3, 0.28, 0.25, 1.0],
            0.5,
            0.8,
            [0.0, 0.0, 0.0],
            0.0,
        ));
        let _chandelier_chains: Vec<MeshId> = CHANDELIER_Z
            .iter()
            .map(|&z| {
                let id = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [0.06, 2.0, 0.06]))).as_mesh().unwrap();
                let _ = v3_demo_common::insert_object(
                    &mut renderer,
                    id,
                    chandelier_mat,
                    glam::Mat4::from_translation(glam::Vec3::new(0.0, 17.5, z)),
                    2.0,
                );
                id
            })
            .collect();
        let _chandelier_rings: Vec<MeshId> = CHANDELIER_Z
            .iter()
            .map(|&z| {
                let id = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [1.2, 0.12, 1.2]))).as_mesh().unwrap();
                let _ = v3_demo_common::insert_object(
                    &mut renderer,
                    id,
                    chandelier_mat,
                    glam::Mat4::from_translation(glam::Vec3::new(0.0, 15.2, z)),
                    1.2,
                );
                id
            })
            .collect();

        // Register lights (chandelier & candle light_ids stored for per-frame flicker updates)
        let mut chandelier_light_ids = Vec::new();
        for &z in CHANDELIER_Z {
            chandelier_light_ids.push(renderer.scene_mut().insert_actor(helio::SceneActor::light(point_light(
                [0.0_f32, 15.0, z],
                [1.0, 0.92, 0.78],
                8.0,
                22.0,
            ))).as_light().unwrap());
        }
        // Stained glass shafts — Stationary: they never animate, so they're excluded
        // from the real-time deferred-light loop once baked lighting is loaded.
        // Without this they were running full tiled PCF every frame despite being "baked".
        for &(x, y, z, r, g, b) in GLASS_LIGHTS {
            let _ = renderer.scene_mut().insert_actor(helio::SceneActor::light_with_movability(
                point_light([x, y, z], [r, g, b], 1.8, 8.0),
                Some(Movability::Stationary),
            ));
        }
        let mut candle_light_ids = Vec::new();
        for &(x, y, z) in CANDLES {
            candle_light_ids.push(renderer.scene_mut().insert_actor(helio::SceneActor::light(point_light(
                [x, y, z],
                [1.0, 0.6, 0.15],
                1.2,
                4.0,
            ))).as_light().unwrap());
        }
        renderer.set_ambient([0.65, 0.7, 0.85], 0.015);
        renderer.set_clear_color([0.0, 0.0, 0.0, 1.0]);

        // Bake static/stationary lights so they're excluded from the real-time
        // deferred-light loop. Without this, all 9 glass window lights + environment
        // run full tiled PCF every frame even though they're fixed.
        renderer.auto_bake(BakeConfig::fast("indoor_cathedral"));

        // ── Decals ───────────────────────────────────────────────────────────
        let mut decal_ids = Vec::new();
        let mut bullet_hole_ids = Vec::new();

        // 1. Worn path on the floor along the nave (multiply blend, dark brown)
        for z in (-24..24).step_by(3) {
            let zf = z as f32;
            decal_ids.push(renderer.scene_mut().insert_actor(
                helio::SceneActor::decal(make_decal(
                    [0.0, 0.02, zf],
                    [0.0, 1.0, 0.0],
                    [1.8, 0.02, 1.2],
                    [0.25, 0.18, 0.12, 0.35],
                    DecalBlendMode::Multiply,
                    DecalType::AlbedoNormal,
                    false,  // normal_adapt: floor is flat, no need
                    0.0,    // persistent
                    0.0,
                )),
            ));
        }

        // 2. Blood splatters near the altar (translucent, red)
        for i in 0..5 {
            let angle = i as f32 * 1.256;
            let r = 0.8 + (i as f32 * 0.3).fract();
            decal_ids.push(renderer.scene_mut().insert_actor(
                helio::SceneActor::decal(make_decal(
                    [(angle.cos() * r) * 1.5, 0.02, -25.0 + (angle.sin() * r) * 1.0],
                    [0.0, 1.0, 0.0],
                    [0.5, 0.02, 0.5],
                    [0.6, 0.02, 0.02, 0.7],
                    DecalBlendMode::AlphaBlend,
                    DecalType::AlbedoNormal,
                    false,
                    0.0,
                    0.0,
                )),
            ));
        }

        // 3. Crack on the left column at z = -6 (normal-only, grey)
        decal_ids.push(renderer.scene_mut().insert_actor(
            helio::SceneActor::decal(make_decal(
                [-5.5, 5.0, -6.0],
                [1.0, 0.0, 0.0],  // facing right (toward nave)
                [0.08, 2.5, 0.3],
                [0.5, 0.5, 0.5, 0.8],
                DecalBlendMode::Translucent,
                DecalType::NormalOnly,
                true,   // normal_adapt: conform to column surface
                0.0,
                0.0,
            )),
        ));

        // 4. Graffiti on left wall (alpha blend, cyan)
        decal_ids.push(renderer.scene_mut().insert_actor(
            helio::SceneActor::decal(make_decal(
                [-10.95, 4.0, 10.0],
                [1.0, 0.0, 0.0],  // facing +X (into the nave)
                [1.0, 1.0, 0.05],
                [0.0, 0.8, 0.8, 0.6],
                DecalBlendMode::AlphaBlend,
                DecalType::AlbedoNormal,
                true,   // normal_adapt: conform to wall surface
                0.0,
                0.0,
            )),
        ));

        // 5. Bullet holes on right wall near entrance (alpha blend, dark grey)
        for i in 0..8 {
            let x = 10.9;
            let y = 2.0 + i as f32 * 0.7;
            let z = 22.0 - (i as f32 * 1.3);
            decal_ids.push(renderer.scene_mut().insert_actor(
                helio::SceneActor::decal(make_decal(
                    [x, y, z],
                    [-1.0, 0.0, 0.0],  // facing -X (into the nave)
                    [0.04, 0.04, 0.04],
                    [0.1, 0.1, 0.1, 0.9],
                    DecalBlendMode::AlphaBlend,
                    DecalType::AlbedoNormal,
                    true,
                    0.0,
                    0.0,
                )),
            ));
            bullet_hole_ids.push(decal_ids[decal_ids.len() - 1]);
        }

        // 6. Soot marks on ceiling under each chandelier (translucent, black)
        for &z in CHANDELIER_Z {
            decal_ids.push(renderer.scene_mut().insert_actor(
                helio::SceneActor::decal(make_decal(
                    [0.0, 20.98, z],
                    [0.0, -1.0, 0.0],  // facing down
                    [1.8, 0.02, 1.8],
                    [0.02, 0.02, 0.02, 0.5],
                    DecalBlendMode::Translucent,
                    DecalType::AlbedoNormal,
                    false,
                    0.0,
                    0.0,
                )),
            ));
        }

        // 7. Glowing rune on the altar (emissive, additive blend, green)
        decal_ids.push(renderer.scene_mut().insert_actor(
            helio::SceneActor::decal(make_decal(
                [0.0, 0.90, -25.5],
                [0.0, 1.0, 0.0],
                [0.6, 0.02, 0.4],
                [0.0, 1.5, 0.3, 0.8],
                DecalBlendMode::Additive,
                DecalType::Emissive,
                false,
                0.0,
                0.0,
            )),
        ));

        // 8. Fading decal near entrance — will disappear after 15 seconds
        let fading_decal_id = Some(renderer.scene_mut().insert_actor(
            helio::SceneActor::decal(make_decal(
                [-10.95, 6.0, 24.0],
                [1.0, 0.0, 0.0],
                [0.8, 0.8, 0.05],
                [1.0, 0.3, 0.1, 1.0],
                DecalBlendMode::AlphaBlend,
                DecalType::AlbedoNormal,
                true,
                8.0,   // fade over 8 seconds
                5.0,   // start fading after 5 seconds
            )),
        ));

        let renderer = Arc::new(Mutex::new(renderer));
        let (bridge, action_rx) = HelioCommandBridge::new();
        let command_bridge = Arc::new(bridge);

        // REPL thread to drive commands from stdin
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
            _floor,
            _nave_ceiling,
            _aisle_ceil_l,
            _aisle_ceil_r,
            _wall_left_outer,
            _wall_right_outer,
            _wall_front,
            _wall_back,
            _colonnade_l,
            _colonnade_r,
            _columns,
            _altar_plinth,
            _altar_step,
            _cross_vert,
            _cross_horiz,
            _pews_left,
            _pews_right,
            _chandelier_chains,
            _chandelier_rings,
            // Start at entrance, looking toward the altar
            cam_pos: glam::Vec3::new(0.0, 2.0, 24.0),
            cam_yaw: std::f32::consts::PI,
            cam_pitch: -0.05,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            debug_mode: 0,
            perf_overlay_mode: PerfOverlayMode::Disabled,
            debug_overlay_enabled: false,
            chandelier_light_ids,
            candle_light_ids,
            start_time: std::time::Instant::now(),
            decal_ids,
            fading_decal_id,
            bullet_hole_ids,
            bullet_hole_spawn_timer: 0.0,
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

            // F1: cycle debug modes (0=normal → 10=shadow heatmap → 11=light-space depth → 0)
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::F1),
                        ..
                    },
                ..
            } => {
                state.debug_mode = match state.debug_mode {
                    0 => 10,
                    10 => 11,
                    _ => 0,
                };
                if let Ok(mut renderer) = state.renderer.lock() {
                    renderer.set_debug_mode(state.debug_mode);
                }
                println!("[debug] shadow debug mode = {}", state.debug_mode);
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::F2),
                        ..
                    },
                ..
            } => {
                state.perf_overlay_mode = match state.perf_overlay_mode {
                    PerfOverlayMode::Disabled => PerfOverlayMode::PassOverdraw,
                    PerfOverlayMode::PassOverdraw => PerfOverlayMode::ShaderComplexity,
                    PerfOverlayMode::ShaderComplexity => PerfOverlayMode::TileLightCount,
                    PerfOverlayMode::TileLightCount => PerfOverlayMode::PassOutput,
                    PerfOverlayMode::PassOutput => PerfOverlayMode::Disabled,
                };
                if let Ok(mut renderer) = state.renderer.lock() {
                    if let Some(pass) = renderer.find_pass_mut::<helio_pass_perf_overlay::PerfOverlayPass>() {
                        pass.set_mode(state.perf_overlay_mode);
                    }
                }
                println!("[debug] perf overlay mode = {:?}", state.perf_overlay_mode);
            }

            // F5: spawn bullet-hole decals near camera
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::F5),
                        ..
                    },
                ..
            } => {
                if let Ok(mut renderer) = state.renderer.lock() {
                    let fwd = glam::Vec3::new(
                        state.cam_yaw.sin() * state.cam_pitch.cos(),
                        state.cam_pitch.sin(),
                        -state.cam_yaw.cos() * state.cam_pitch.cos(),
                    );
                    for i in 0..5 {
                        let spread = 0.3;
                        let offset = glam::Vec3::new(
                            (i as f32 * 0.7 - 1.4) * spread,
                            (i as f32 * 0.3) * spread,
                            0.0,
                        );
                        let hit_pos = state.cam_pos + fwd * 3.0 + offset;
                        let decal = make_decal(
                            [hit_pos.x, hit_pos.y, hit_pos.z],
                            [fwd.x, fwd.y, fwd.z],
                            [0.04, 0.04, 0.04],
                            [0.08, 0.08, 0.08, 0.9],
                            DecalBlendMode::AlphaBlend,
                            DecalType::AlbedoNormal,
                            true,
                            0.0,
                            0.0,
                        );
                        let id = renderer
                            .scene_mut()
                            .insert_actor(helio::SceneActor::decal(decal))
                            .as_decal()
                            .unwrap();
                        state.bullet_hole_ids.push(id);
                        state.decal_ids.push(id);
                    }
                    println!("[decals] spawned 5 bullet holes");
                }
            }

            // F3: toggle debug overlay (FPS, timings, texture stats)
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::F3),
                        ..
                    },
                ..
            } => {
                state.debug_overlay_enabled = !state.debug_overlay_enabled;
                if let Ok(mut renderer) = state.renderer.lock() {
                    if let Some(pass) = renderer.find_pass_mut::<helio_pass_debug_overlay::DebugOverlayPass>() {
                        pass.set_enabled(state.debug_overlay_enabled);
                    }
                }
                println!("[debug] debug overlay = {}", state.debug_overlay_enabled);
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
        const SPEED: f32 = 5.0;
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

        let size = self.window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;
        let time = self.start_time.elapsed().as_secs_f32();

        let camera = Camera::perspective_look_at(
            self.cam_pos,
            self.cam_pos + forward,
            glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            aspect,
            0.1,
            200.0,
        );

        // Apply commands from REPL / quark to renderer
        let mut renderer = self.renderer.lock().unwrap();
        while let Ok(action) = self.action_rx.try_recv() {
            match action {
                HelioAction::SetDebugMode(mode) => renderer.set_debug_mode(mode),
                HelioAction::SetEditorMode(enabled) => renderer.set_editor_mode(enabled),
                HelioAction::DebugClear => renderer.debug_clear(),
            }
        }

        // Chandeliers flicker slightly
        let flicker = 1.0 + (time * 9.1).sin() * 0.03 + (time * 5.7).cos() * 0.02;
        // Candle flicker — more pronounced
        let cflicker = 1.0 + (time * 14.3).sin() * 0.07 + (time * 8.9).cos() * 0.05;

        // Update flickering chandelier intensities
        for (i, &id) in self.chandelier_light_ids.iter().enumerate() {
            let z = CHANDELIER_Z[i];
            let _ = renderer.scene_mut().update_light(
                id,
                point_light([0.0_f32, 15.0, z], [1.0, 0.92, 0.78], 8.0 * flicker, 22.0),
            );
        }
        // Update flickering candle intensities
        for (i, &id) in self.candle_light_ids.iter().enumerate() {
            let (x, y, z) = CANDLES[i];
            let _ = renderer.scene_mut().update_light(
                id,
                point_light([x, y, z], [1.0, 0.6, 0.15], 1.2 * cflicker, 4.0),
            );
        }

        // ── Update fading decal age ──────────────────────────────────────
        if let Some(id) = self.fading_decal_id {
            let mut renderer = self.renderer.lock().unwrap();
            // Read current decal, advance age, write back
            // We reconstruct the decal on the app side since Scene::decals is crate-internal
            let decal = make_decal(
                [-10.95, 6.0, 24.0],
                [1.0, 0.0, 0.0],
                [0.8, 0.8, 0.05],
                [1.0, 0.3, 0.1, 1.0],
                DecalBlendMode::AlphaBlend,
                DecalType::AlbedoNormal,
                true,
                8.0,
                5.0,
            );
            // We need to track the age separately since we can't read the GPU value
            // For simplicity, reconstruct with an incrementing age
            // In a real app, you'd store ages on CPU or expose decal read access
            let age = self.start_time.elapsed().as_secs_f32() - 0.0; // Start at zero
            let updated = GpuDecal {
                age,
                ..decal
            };
            renderer.scene_mut().update_decal(id, updated);
        }

        // Auto-spawn bullet holes periodically
        self.bullet_hole_spawn_timer += dt;
        if self.bullet_hole_spawn_timer > 3.0 {
            self.bullet_hole_spawn_timer = 0.0;
            let mut renderer = self.renderer.lock().unwrap();
            let fwd = glam::Vec3::new(
                self.cam_yaw.sin() * self.cam_pitch.cos(),
                self.cam_pitch.sin(),
                -self.cam_yaw.cos() * self.cam_pitch.cos(),
            );
            let right = glam::Vec3::new(self.cam_yaw.cos(), 0.0, self.cam_yaw.sin());
            let hit_pos = self.cam_pos + fwd * 4.0 + right * 1.5;
            let decal = make_decal(
                [hit_pos.x, hit_pos.y, hit_pos.z],
                [fwd.x, fwd.y, fwd.z],
                [0.04, 0.04, 0.04],
                [0.08, 0.08, 0.08, 0.9],
                DecalBlendMode::AlphaBlend,
                DecalType::AlbedoNormal,
                true,
                0.0,
                0.0,
            );
            let id = renderer
                .scene_mut()
                .insert_actor(helio::SceneActor::decal(decal))
                .as_decal()
                .unwrap();
            self.decal_ids.push(id);
        }

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) => t,
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => {
                log::warn!("surface acquire failed");
                return;
            }
        };
        let view = output.texture.create_view(&Default::default());

        if let Err(e) = renderer.render(&camera, &view) {
            log::error!("Render: {:?}", e);
        }
        renderer.queue().present(output);
    }
}



