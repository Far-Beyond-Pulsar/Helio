//! Indoor cathedral example – high complexity
//!
//! A large Gothic cathedral interior: a 60 m nave flanked by two side aisles,
//! 12 stone columns, a raised altar platform with a cross, carved stone pews
//! in 6 rows on each side, three ornate chandeliers, stained-glass window
//! shafts casting coloured light at intervals along both walls, and candle
//! clusters near the altar.
//!
//! No sky atmosphere — the scene relies entirely on the interplay of the
//! chandelier warm-white lights, the cool-coloured stained-glass fills, and
//! a very dim stone-cold ambient to create a moody sacred atmosphere. The
//! radiance cascades GI system bounces chandelier light deep into the side
//! aisles and onto the vaulted ceiling.
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Mouse drag  — look around (click to grab cursor)
//!   Escape      — release cursor / exit

mod demo_portal;
mod v3_demo_common;

use helio::{
    required_wgpu_features, required_wgpu_limits, Camera, HelioAction, HelioCommandBridge, LightId,
    MeshId, ObjectId, Renderer, RendererConfig, WaterHitboxDescriptor, WaterHitboxId,
};
use v3_demo_common::{
    box_mesh, insert_object, make_material, plane_mesh, point_light, sphere_mesh,
};

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

/// Build a WaterHitboxDescriptor for a sphere at `new_pos` (previously at `old_pos`).
///
/// The hitbox coordinate system is: X/Z normalized to [-1,1] over the pool half-extent,
/// Y is relative to the water surface (0 = surface, negative = submerged).
fn ball_aabb(
    old_pos: glam::Vec3,
    new_pos: glam::Vec3,
    radius: f32,
    surface_y: f32,
    pool_half_xz: f32,
) -> WaterHitboxDescriptor {
    let to_sim = |p: glam::Vec3| -> ([f32; 3], [f32; 3]) {
        let mn = [
            (p.x - radius) / pool_half_xz,
            (p.y - radius) - surface_y,
            (p.z - radius) / pool_half_xz,
        ];
        let mx = [
            (p.x + radius) / pool_half_xz,
            (p.y + radius) - surface_y,
            (p.z + radius) / pool_half_xz,
        ];
        (mn, mx)
    };
    let (old_min, old_max) = to_sim(old_pos);
    let (new_min, new_max) = to_sim(new_pos);
    WaterHitboxDescriptor {
        old_min,
        old_max,
        new_min,
        new_max,
        edge_softness: 0.15,
        strength: 1.2,
    }
}

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

    // Ball physics
    ball_obj_id: ObjectId,
    ball_hitbox_id: WaterHitboxId,
    ball_pos: glam::Vec3,
    ball_vel: glam::Vec3,
    ball_prev_pos: glam::Vec3,

    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    // Debug
    debug_mode: u32,

    // Scene state
    chandelier_light_ids: Vec<LightId>,
    candle_light_ids: Vec<LightId>,
    start_time: std::time::Instant,
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

        let mut renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            RendererConfig::new(size.width, size.height, format)
                .with_shadow_quality(helio::ShadowQuality::Ultra),
        );
        renderer.set_editor_mode(true);

        // Start live performance portal
        demo_portal::enable_live_dashboard(&mut renderer);

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

        // === AAA QUALITY WATER POOL ===
        // Realistic oceanographic parameters for stunning photorealistic water
        let pool = helio::WaterVolumeDescriptor {
            bounds_min: [-6.0, 0.3, -6.0], // 12x12 meter pool, slightly raised
            bounds_max: [6.0, 2.5, 6.0],   // 2.2m deep pool
            surface_height: 1.8,           // Water surface at 1.8m above floor

            // GERSTNER WAVE PARAMETERS (realistic ocean physics)
            wave_amplitude: 0.12, // Realistic wave height (12cm primary waves)
            wave_frequency: 1.2,  // Higher frequency for detailed ripples
            wave_speed: 1.5,      // Natural wave propagation speed
            wave_direction: [0.7, 0.4], // Diagonal wave direction for natural look
            wave_steepness: 0.65, // Sharp, realistic wave peaks (0.6-0.7 is physically accurate)

            // WATER OPTICAL PROPERTIES (crystal clear pool water)
            water_color: [0.05, 0.20, 0.30], // Light blue-green for clear water
            extinction: [0.08, 0.04, 0.02], // Very low absorption for crystal clear water (reduced by ~55%)

            // FOAM PARAMETERS (white caps on wave crests)
            foam_threshold: 0.68, // Foam appears on steeper waves
            foam_amount: 0.75,    // Generous foam coverage for realism

            // REFLECTION & REFRACTION (physically accurate)
            reflection_strength: 1.0, // Full reflectivity (water is ~100% reflective at grazing angles)
            refraction_strength: 1.0, // Full refraction strength for chromatic aberration
            fresnel_power: 5.0,       // Physically-based fresnel (water IOR ~1.333)

            // CAUSTICS (underwater light patterns - VISIBLE)
            caustics_enabled: true,
            caustics_intensity: 4.0, // Strong visible caustics
            caustics_scale: 8.0,     // Larger, more visible patterns
            caustics_speed: 1.5,     // Faster animation for visibility

            // VOLUMETRIC EFFECTS (subtle for clear visibility)
            fog_density: 0.015, // Very subtle underwater fog (reduced from 0.06)
            god_rays_intensity: 0.2, // Subtle god rays through water (reduced from 2.0)
            // Sim-based rendering parameters (defaults: IOR 1.333, physically-based fresnel)
            ..Default::default()
        };
        renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::water_volume(pool));

        // === BOUNCING BALL ===
        // A shiny sphere that bounces perfectly on the water surface, creating ripple waves.
        const BALL_RADIUS: f32 = 0.5;
        const WATER_SURFACE: f32 = 1.8;
        const POOL_HALF_XZ: f32 = 6.0;

        let ball_start = glam::Vec3::new(0.0, WATER_SURFACE + 4.0, 0.0);
        // Give it a diagonal horizontal kick so it travels across the pool
        let ball_start_vel = glam::Vec3::new(1.8, 0.0, 1.2);
        let ball_mesh_id = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(sphere_mesh(
                [0.0, 0.0, 0.0],
                BALL_RADIUS,
            )))
            .as_mesh()
            .unwrap();
        let ball_mat = renderer.scene_mut().insert_material(make_material(
            [0.9, 0.15, 0.1, 1.0], // red
            0.2,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));
        let ball_obj_id = insert_object(
            &mut renderer,
            ball_mesh_id,
            ball_mat,
            glam::Mat4::from_translation(ball_start),
            BALL_RADIUS,
        )
        .expect("ball object");

        let ball_hitbox_id = renderer
            .scene_mut()
            .insert_water_hitbox(ball_aabb(
                ball_start,
                ball_start,
                BALL_RADIUS,
                WATER_SURFACE,
                POOL_HALF_XZ,
            ))
            .expect("ball hitbox");

        // Nave + aisles: total width = 22m (x: -11..+11), length = 60m (z: -28..+28), height = 21m
        // Expand floor to cover full cathedral footprint. 32m radius = 64m square.
        let _floor = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(plane_mesh([0.0, 0.0, 0.0], 32.0)))
            .as_mesh()
            .unwrap();
        let _wall_back = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(box_mesh(
                [0.0, 0.0, 0.0],
                [11.0, 10.5, 0.25],
            )))
            .as_mesh()
            .unwrap();
        let _wall_front = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(box_mesh(
                [0.0, 0.0, 0.0],
                [11.0, 10.5, 0.25],
            )))
            .as_mesh()
            .unwrap();
        let _aisle_ceil_l = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(box_mesh(
                [0.0, 0.0, 0.0],
                [2.5, 0.15, 28.0],
            )))
            .as_mesh()
            .unwrap();
        let _nave_ceiling = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(box_mesh(
                [0.0, 0.0, 0.0],
                [6.0, 0.18, 28.0],
            )))
            .as_mesh()
            .unwrap();
        let _aisle_ceil_r = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(box_mesh(
                [0.0, 0.0, 0.0],
                [2.5, 0.15, 28.0],
            )))
            .as_mesh()
            .unwrap();
        let _wall_left_outer = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(box_mesh(
                [0.0, 0.0, 0.0],
                [0.25, 7.0, 28.0],
            )))
            .as_mesh()
            .unwrap();
        let _wall_right_outer = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(box_mesh(
                [0.0, 0.0, 0.0],
                [0.25, 7.0, 28.0],
            )))
            .as_mesh()
            .unwrap();
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
                let id = renderer
                    .scene_mut()
                    .insert_actor(helio::SceneActor::mesh(box_mesh(
                        [0.0, 0.0, 0.0],
                        [0.25, 5.5, half_len.max(0.1)],
                    )))
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
                let id = renderer
                    .scene_mut()
                    .insert_actor(helio::SceneActor::mesh(box_mesh(
                        [0.0, 0.0, 0.0],
                        [0.25, 5.5, half_len.max(0.1)],
                    )))
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
                let l = renderer
                    .scene_mut()
                    .insert_actor(helio::SceneActor::mesh(box_mesh(
                        [0.0, 0.0, 0.0],
                        [0.65, 10.0, 0.65],
                    )))
                    .as_mesh()
                    .unwrap();
                let _ = v3_demo_common::insert_object(
                    &mut renderer,
                    l,
                    mat,
                    glam::Mat4::from_translation(glam::Vec3::new(-5.5, 10.0, z)),
                    10.0,
                );
                let r = renderer
                    .scene_mut()
                    .insert_actor(helio::SceneActor::mesh(box_mesh(
                        [0.0, 0.0, 0.0],
                        [0.65, 10.0, 0.65],
                    )))
                    .as_mesh()
                    .unwrap();
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
        let _altar_step = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(box_mesh(
                [0.0, 0.0, 0.0],
                [5.5, 0.20, 3.0],
            )))
            .as_mesh()
            .unwrap();
        let _altar_plinth = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(box_mesh(
                [0.0, 0.0, 0.0],
                [3.0, 0.45, 1.5],
            )))
            .as_mesh()
            .unwrap();
        let _cross_vert = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(box_mesh(
                [0.0, 0.0, 0.0],
                [0.18, 2.2, 0.18],
            )))
            .as_mesh()
            .unwrap();
        let _cross_horiz = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(box_mesh(
                [0.0, 0.0, 0.0],
                [1.0, 0.18, 0.18],
            )))
            .as_mesh()
            .unwrap();
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
                let id = renderer
                    .scene_mut()
                    .insert_actor(helio::SceneActor::mesh(box_mesh(
                        [0.0, 0.0, 0.0],
                        [1.5, 0.45, 0.5],
                    )))
                    .as_mesh()
                    .unwrap();
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
                let id = renderer
                    .scene_mut()
                    .insert_actor(helio::SceneActor::mesh(box_mesh(
                        [0.0, 0.0, 0.0],
                        [1.5, 0.45, 0.5],
                    )))
                    .as_mesh()
                    .unwrap();
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
                let id = renderer
                    .scene_mut()
                    .insert_actor(helio::SceneActor::mesh(box_mesh(
                        [0.0, 0.0, 0.0],
                        [0.06, 2.0, 0.06],
                    )))
                    .as_mesh()
                    .unwrap();
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
                let id = renderer
                    .scene_mut()
                    .insert_actor(helio::SceneActor::mesh(box_mesh(
                        [0.0, 0.0, 0.0],
                        [1.2, 0.12, 1.2],
                    )))
                    .as_mesh()
                    .unwrap();
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
            chandelier_light_ids.push(
                renderer
                    .scene_mut()
                    .insert_actor(helio::SceneActor::light(point_light(
                        [0.0_f32, 15.0, z],
                        [1.0, 0.92, 0.78],
                        8.0,
                        22.0,
                    )))
                    .as_light()
                    .unwrap(),
            );
        }
        // Stained glass shafts — static, no need to store ids
        for &(x, y, z, r, g, b) in GLASS_LIGHTS {
            let _ = renderer
                .scene_mut()
                .insert_actor(helio::SceneActor::light(point_light(
                    [x, y, z],
                    [r, g, b],
                    1.8,
                    8.0,
                )));
        }
        let mut candle_light_ids = Vec::new();
        for &(x, y, z) in CANDLES {
            candle_light_ids.push(
                renderer
                    .scene_mut()
                    .insert_actor(helio::SceneActor::light(point_light(
                        [x, y, z],
                        [1.0, 0.6, 0.15],
                        1.2,
                        4.0,
                    )))
                    .as_light()
                    .unwrap(),
            );
        }
        renderer.set_ambient([0.65, 0.7, 0.85], 0.015);
        renderer.set_clear_color([0.0, 0.0, 0.0, 1.0]);

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
            ball_obj_id,
            ball_hitbox_id,
            ball_pos: ball_start,
            ball_vel: ball_start_vel,
            ball_prev_pos: ball_start,
            // Start at entrance, looking toward the altar
            cam_pos: glam::Vec3::new(0.0, 2.0, 24.0),
            cam_yaw: std::f32::consts::PI,
            cam_pitch: -0.05,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            debug_mode: 0,
            chandelier_light_ids,
            candle_light_ids,
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
                HelioAction::SetDebugDepthTest(enabled) => renderer.set_debug_depth_test(enabled),
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

        // Scene state is persistent — no per-frame setup needed.

        // ---- Bouncing ball physics -------------------------------------------
        const GRAVITY: f32 = -9.8;
        const BALL_RADIUS: f32 = 0.5;
        const WATER_SURFACE: f32 = 1.8;
        const POOL_HALF_XZ: f32 = 6.0;

        let prev_pos = self.ball_pos;
        self.ball_vel.y += GRAVITY * dt;
        self.ball_pos += self.ball_vel * dt;

        // Perfect elastic bounce off water — restores full height every time
        let floor_y = WATER_SURFACE + BALL_RADIUS;
        if self.ball_pos.y < floor_y {
            self.ball_pos.y = floor_y;
            self.ball_vel.y = self.ball_vel.y.abs(); // no energy loss on vertical
        }

        // Elastic bounce off pool walls (no energy loss)
        let limit = POOL_HALF_XZ - BALL_RADIUS;
        if self.ball_pos.x.abs() > limit {
            self.ball_pos.x = self.ball_pos.x.signum() * limit;
            self.ball_vel.x = -self.ball_vel.x;
        }
        if self.ball_pos.z.abs() > limit {
            self.ball_pos.z = self.ball_pos.z.signum() * limit;
            self.ball_vel.z = -self.ball_vel.z;
        }

        let _ = renderer.scene_mut().update_object_transform(
            self.ball_obj_id,
            glam::Mat4::from_translation(self.ball_pos),
        );
        let _ = renderer.scene_mut().update_water_hitbox(
            self.ball_hitbox_id,
            ball_aabb(
                prev_pos,
                self.ball_pos,
                BALL_RADIUS,
                WATER_SURFACE,
                POOL_HALF_XZ,
            ),
        );
        self.ball_prev_pos = prev_pos;
        // ---------------------------------------------------------------------

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
