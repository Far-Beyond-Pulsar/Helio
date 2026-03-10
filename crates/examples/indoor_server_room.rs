//! Indoor server room example – high complexity
//!
//! A large data-centre floor: four rows of eight server racks each (32 racks
//! total), cold-aisle / hot-aisle separation walls, overhead cable trays,
//! eight ceiling fluorescent panel lights, four rear-wall cooling units, and
//! per-row status LED lighting (green = healthy, amber = warning, red = alert).
//!
//! The very cool near-UV ambient and tightly spaced overhead panels create
//! the distinctive blue-white clinical look of a real data centre. Bloom is
//! pushed harder than other scenes to give the status LEDs a sharp halo.
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Mouse drag  — look around (click to grab cursor)
//!   Escape      — release cursor / exit



mod demo_portal;

use helio_render_v2::{Renderer, RendererConfig, Camera, GpuMesh, SceneLight, LightId, BillboardId};

use helio_render_v2::features::{
    FeatureRegistry, LightingFeature, BloomFeature, ShadowsFeature,
    BillboardsFeature, BillboardInstance, RadianceCascadesFeature,
};


use winit::{
    application::ApplicationHandler, event::*, event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey}, window::{Window, WindowId, CursorGrabMode},
};


use std::collections::HashSet;


use std::sync::Arc;

// ── Scene data ────────────────────────────────────────────────────────────────

// Rack rows: (center_x, rack_color_tag)
// 0=green, 1=green, 2=amber, 3=red
const RACK_ROWS: &[(f32, u8)] = &[
    (-7.5, 0),
    (-2.5, 1),
    ( 2.5, 2),
    ( 7.5, 3),
];

// 8 racks per row, spaced 1.2 m apart centered on Z=0 (−4.2 … +4.2)
const RACK_Z_OFFSETS: &[f32] = &[-4.2, -3.0, -1.8, -0.6, 0.6, 1.8, 3.0, 4.2];

// Overhead fluorescent panels: 2 per row at y≈3.8, one toward each end
const CEILING_PANEL_XZ: &[(f32, f32)] = &[
    (-7.5, -3.5), (-7.5, 3.5),
    (-2.5, -3.5), (-2.5, 3.5),
    ( 2.5, -3.5), ( 2.5, 3.5),
    ( 7.5, -3.5), ( 7.5, 3.5),
];

// Status light colors per tag
fn row_color(tag: u8) -> [f32; 3] {
    match tag {
        0 => [0.0, 1.0, 0.2],    // healthy green
        1 => [0.0, 0.9, 0.5],    // green-teal
        2 => [1.0, 0.65, 0.0],   // amber warning
        _ => [1.0, 0.05, 0.05],  // critical red
    }
}

// ─────────────────────────────────────────────────────────────────────────────

fn load_sprite() -> (Vec<u8>, u32, u32) {
    let img = image::load_from_memory(include_bytes!("../../spotlight.png"))
        .unwrap_or_else(|_| image::DynamicImage::new_rgba8(1, 1))
        .into_rgba8();
    let (w, h) = img.dimensions();
    (img.into_raw(), w, h)
}

fn load_probe_sprite() -> (Vec<u8>, u32, u32) {
    let img = image::load_from_memory(include_bytes!("../../probe.png"))
        .unwrap_or_else(|_| image::DynamicImage::new_rgba8(1, 1))
        .into_rgba8();
    let (w, h) = img.dimensions();
    (img.into_raw(), w, h)
}

const RC_WORLD_MIN: [f32; 3] = [-12.0, -0.1, -8.0];
const RC_WORLD_MAX: [f32; 3] = [12.0, 5.0, 8.0];

fn probe_billboards(world_min: [f32; 3], world_max: [f32; 3]) -> Vec<helio_render_v2::features::BillboardInstance> {
    use helio_render_v2::features::radiance_cascades::PROBE_DIMS;
    const COLORS: [[f32; 4]; 4] = [
        [0.0, 1.0, 1.0, 0.85],
        [0.0, 1.0, 0.0, 0.80],
        [1.0, 1.0, 0.0, 0.75],
        [1.0, 0.35, 0.0, 0.70],
    ];
    // screen_scale=true: sizes are angular (multiplied by distance), giving constant apparent size
    const SIZES: [[f32; 2]; 4] = [
        [0.035, 0.035],  // cascade 0 — finest (4096 probes) — tiny dots
        [0.075, 0.075],  // cascade 1
        [0.140, 0.140],  // cascade 2
        [0.260, 0.260],  // cascade 3 — coarsest (8 probes) — large markers
    ];
    let mut out = Vec::new();
    for (c, &dim) in PROBE_DIMS.iter().enumerate() {
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    let x = world_min[0] + (i as f32 + 0.5) / dim as f32 * (world_max[0] - world_min[0]);
                    let y = world_min[1] + (j as f32 + 0.5) / dim as f32 * (world_max[1] - world_min[1]);
                    let z = world_min[2] + (k as f32 + 0.5) / dim as f32 * (world_max[2] - world_min[2]);
                    out.push(helio_render_v2::features::BillboardInstance::new([x, y, z], SIZES[c])
                        .with_color(COLORS[c])
                        .with_screen_scale(true));
                }
            }
        }
    }
    out
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run");
}

struct App { state: Option<AppState> }

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    surface_format: wgpu::TextureFormat,
    renderer: Renderer,
    last_frame: std::time::Instant,

    floor:   GpuMesh,
    ceiling: GpuMesh,
    wall_n:  GpuMesh,
    wall_s:  GpuMesh,
    wall_e:  GpuMesh,
    wall_w:  GpuMesh,
    // Raised floor tiles (cable access panels)
    floor_tiles: Vec<GpuMesh>,
    // Server racks (4 rows × 8 racks = 32)
    racks: Vec<GpuMesh>,
    // Hot-aisle containment walls between rack rows
    hot_aisle_walls: Vec<GpuMesh>,
    // Overhead cable trays (one per rack row)
    cable_trays: Vec<GpuMesh>,
    // Ceiling fluorescent panel bodies
    ceiling_panels: Vec<GpuMesh>,
    // Rear-wall cooling units
    cooling_units: Vec<GpuMesh>,
    // Entry door alcove
    door_frame: GpuMesh,
    door_panel: GpuMesh,

    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    probe_vis: bool,
    sprite_w: u32,
    sprite_h: u32,

    // Persistent light handles — registered once, never rebuilt per-frame.
    _light_ids: Vec<LightId>,
    // Persistent billboard handles — swapped between spotlight and probe views.
    billboard_ids: Vec<BillboardId>,
    // Pre-computed probe billboard data so toggling doesn't allocate.
    spotlight_billboards: Vec<BillboardInstance>,
    probe_billboard_data: Vec<BillboardInstance>,
}

impl App {
    fn new() -> Self { Self { state: None } }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }

        let window = Arc::new(event_loop.create_window(
            Window::default_attributes()
                .with_title("Helio – Indoor Server Room")
                .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
        ).expect("window"));

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::all(), ..Default::default() });
        let surface  = instance.create_surface(window.clone()).expect("surface");
        let adapter  = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface), force_fallback_adapter: false,
        })).expect("adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY | wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
            required_limits: wgpu::Limits::default().using_minimum_supported_acceleration_structure_values(),
            memory_hints: wgpu::MemoryHints::default(),
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            trace: wgpu::Trace::Off,
        })).expect("device");
        let device = Arc::new(device);
        let queue  = Arc::new(queue);

        let caps   = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().find(|f| f.is_srgb()).copied().unwrap_or(caps.formats[0]);
        let size   = window.inner_size();
        surface.configure(&device, &wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format, width: size.width, height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![], desired_maximum_frame_latency: 2,
        });

        let (sprite_rgba, sprite_w, sprite_h) = load_sprite();
        let features = FeatureRegistry::builder()
            .with_feature(LightingFeature::new())
            .with_feature(BloomFeature::new().with_intensity(0.7).with_threshold(0.8))
            .with_feature(ShadowsFeature::new().with_atlas_size(1024).with_max_lights(4))
            .with_feature(BillboardsFeature::new().with_sprite(sprite_rgba, sprite_w, sprite_h).with_max_instances(5000))
            .with_feature(RadianceCascadesFeature::new()
                .with_world_bounds([-12.0, -0.1, -8.0], [12.0, 5.0, 8.0]))
            .build();

        let mut renderer = Renderer::new(device.clone(), queue.clone(),
            RendererConfig::new(size.width, size.height, format, features),
        ).expect("renderer");

        // start live portal so snapshots (draw counts, timings, scene layout) are
        // published when the app runs.  Without this call `live_portal` remains
        // `None` and the web UI will stay empty ("still 0" was caused by this).
        match renderer.start_live_portal_default() {
            Ok(url) => log::info!("Helio live portal: {url}"),
            Err(e)  => log::warn!("Could not start live portal: {e}"),
        }

        // Room: 24 m wide (X: -12..+12), 12 m deep (Z: -6..+6), 4 m tall
        let floor   = renderer.create_mesh_plane([0.0, 0.0, 0.0], 12.0);
        let ceiling = renderer.create_mesh_rect3d([0.0, 4.0, 0.0], [12.0, 0.05, 6.0]);
        let wall_n  = renderer.create_mesh_rect3d([ 0.0, 2.0, -6.0], [12.0, 2.0, 0.05]);
        let wall_s  = renderer.create_mesh_rect3d([ 0.0, 2.0,  6.0], [12.0, 2.0, 0.05]);
        let wall_e  = renderer.create_mesh_rect3d([ 12.0, 2.0, 0.0], [0.05, 2.0, 6.0]);
        let wall_w  = renderer.create_mesh_rect3d([-12.0, 2.0, 0.0], [0.05, 2.0, 6.0]);

        // Raised floor tiles (cable trench access): thin slabs arranged in a grid
        let mut floor_tiles: Vec<GpuMesh> = Vec::new();
        for xi in -2_i32..=2 {
            for zi in -1_i32..=1 {
                floor_tiles.push(renderer.create_mesh_rect3d(
                    [xi as f32 * 4.0, 0.03, zi as f32 * 3.5],
                    [1.9, 0.03, 1.7]));
            }
        }

        // Racks: 2 m tall, 0.6 m wide, 0.9 m deep
        let mut racks: Vec<GpuMesh> = Vec::new();
        for &(rx, _) in RACK_ROWS {
            for &rz in RACK_Z_OFFSETS {
                racks.push(renderer.create_mesh_rect3d([rx, 1.0, rz], [0.3, 1.0, 0.45]));
            }
        }

        // Hot-aisle containment walls: thin vertical slabs between row pairs
        // Between rows 0-1 (-5.0 x) and rows 2-3 (+5.0 x)
        let hot_aisle_walls = vec![
            renderer.create_mesh_rect3d([-5.0, 1.5, 0.0], [0.05, 1.5, 5.0]),
            renderer.create_mesh_rect3d([ 5.0, 1.5, 0.0], [0.05, 1.5, 5.0]),
        ];

        // Cable trays: one per row, running full depth overhead
        let cable_trays: Vec<GpuMesh> = RACK_ROWS.iter().map(|&(rx, _)| {
            renderer.create_mesh_rect3d([rx, 3.55, 0.0], [0.25, 0.08, 5.5])
        }).collect();

        // Ceiling fluorescent panel bodies
        let ceiling_panels: Vec<GpuMesh> = CEILING_PANEL_XZ.iter().map(|&(px, pz)| {
            renderer.create_mesh_rect3d([px, 3.92, pz], [0.3, 0.04, 0.8])
        }).collect();

        // Rear cooling units: 4 units on north wall (z = -5.8), tall and wide
        let cooling_units: Vec<GpuMesh> = [-9.0_f32, -3.0, 3.0, 9.0].iter().map(|&cx| {
            renderer.create_mesh_rect3d([cx, 1.5, -5.75], [1.1, 1.5, 0.25])
        }).collect();

        // Entry door on south wall (z = +6): frame + recessed panel
        let door_frame = renderer.create_mesh_rect3d([0.0, 1.2, 5.9], [0.7, 1.2, 0.08]);
        let door_panel = renderer.create_mesh_rect3d([0.0, 1.0, 5.95], [0.55, 1.0, 0.04]);
        demo_portal::enable_live_dashboard(&mut renderer);

        renderer.add_object(&floor,      None, glam::Mat4::IDENTITY);
        renderer.add_object(&ceiling,    None, glam::Mat4::IDENTITY);
        renderer.add_object(&wall_n,     None, glam::Mat4::IDENTITY);
        renderer.add_object(&wall_s,     None, glam::Mat4::IDENTITY);
        renderer.add_object(&wall_e,     None, glam::Mat4::IDENTITY);
        renderer.add_object(&wall_w,     None, glam::Mat4::IDENTITY);
        renderer.add_object(&door_frame, None, glam::Mat4::IDENTITY);
        renderer.add_object(&door_panel, None, glam::Mat4::IDENTITY);
        for m in &floor_tiles      { renderer.add_object(m, None, glam::Mat4::IDENTITY); }
        for m in &racks            { renderer.add_object(m, None, glam::Mat4::IDENTITY); }
        for m in &hot_aisle_walls  { renderer.add_object(m, None, glam::Mat4::IDENTITY); }
        for m in &cable_trays      { renderer.add_object(m, None, glam::Mat4::IDENTITY); }
        for m in &ceiling_panels   { renderer.add_object(m, None, glam::Mat4::IDENTITY); }
        for m in &cooling_units    { renderer.add_object(m, None, glam::Mat4::IDENTITY); }

        // ── Register lights once — no per-frame Vec allocation ────────────
        let mut light_ids: Vec<LightId> = Vec::new();
        // Overhead fluorescent panels
        for &(px, pz) in CEILING_PANEL_XZ {
            light_ids.push(renderer.add_light(SceneLight::spot(
                [px, 3.78, pz], [0.0, -1.0, 0.0],
                [0.88, 0.93, 1.0], 4.5, 7.0, 1.22, 1.48,
            )));
        }
        // Per-row status LED strips
        for &(rx, tag) in RACK_ROWS {
            let col = row_color(tag);
            light_ids.push(renderer.add_light(SceneLight::point([rx, 2.1, 0.0], col, 2.5, 6.0)));
            for &end_z in &[-4.5_f32, 4.5] {
                light_ids.push(renderer.add_light(SceneLight::point([rx, 2.1, end_z], col, 1.0, 3.5)));
            }
        }
        // Cooling unit indicator lights
        for (i, cx) in [-9.0_f32, -3.0, 3.0, 9.0].iter().enumerate() {
            let col: [f32; 3] = if i == 1 { [0.0, 1.0, 0.3] } else { [0.0, 0.6, 1.0] };
            light_ids.push(renderer.add_light(SceneLight::point([*cx, 2.8, -5.6], col, 0.8, 3.0)));
        }

        // ── Ambient (set once, unchanged) ─────────────────────────────────
        renderer.set_ambient([0.6, 0.72, 1.0], 0.06);

        // ── Billboards: build both sets, register spotlight set first ─────
        let spotlight_billboards: Vec<BillboardInstance> = {
            let mut bb = Vec::new();
            for &(px, pz) in CEILING_PANEL_XZ {
                bb.push(BillboardInstance::new([px, 3.78, pz], [0.35, 0.12]).with_color([0.88, 0.93, 1.0, 1.0]));
            }
            for &(rx, tag) in RACK_ROWS {
                let col = row_color(tag);
                bb.push(BillboardInstance::new([rx, 2.1, 0.0], [0.12, 0.08]).with_color([col[0], col[1], col[2], 1.0]));
                for &end_z in &[-4.5_f32, 4.5] {
                    bb.push(BillboardInstance::new([rx, 2.1, end_z], [0.1, 0.08]).with_color([col[0], col[1], col[2], 0.9]));
                }
            }
            for (i, cx) in [-9.0_f32, -3.0, 3.0, 9.0].iter().enumerate() {
                let col: [f32; 3] = if i == 1 { [0.0, 1.0, 0.3] } else { [0.0, 0.6, 1.0] };
                bb.push(BillboardInstance::new([*cx, 2.8, -5.6], [0.12, 0.08]).with_color([col[0], col[1], col[2], 0.85]));
            }
            bb
        };
        let probe_billboard_data = probe_billboards(RC_WORLD_MIN, RC_WORLD_MAX);

        // Register the initial (spotlight) set and keep handles.
        let mut billboard_ids: Vec<BillboardId> = Vec::with_capacity(spotlight_billboards.len());
        for inst in &spotlight_billboards {
            billboard_ids.push(renderer.add_billboard(inst.clone()));
        }

        self.state = Some(AppState {
            window, surface, device, surface_format: format, renderer,
            last_frame: std::time::Instant::now(),
            floor, ceiling, wall_n, wall_s, wall_e, wall_w,
            floor_tiles, racks, hot_aisle_walls, cable_trays,
            ceiling_panels, cooling_units, door_frame, door_panel,
            // Start at the door end of the room, looking into the server rows
            cam_pos: glam::Vec3::new(0.0, 1.75, 5.0),
            cam_yaw: std::f32::consts::PI, cam_pitch: -0.05,
            keys: HashSet::new(), cursor_grabbed: false, mouse_delta: (0.0, 0.0),
            probe_vis: false,
            sprite_w,
            sprite_h,
            _light_ids: light_ids,
            billboard_ids,
            spotlight_billboards,
            probe_billboard_data,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event: KeyEvent {
                state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::Escape), ..
            }, .. } => {
                if state.cursor_grabbed {
                    state.cursor_grabbed = false;
                    let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                    state.window.set_cursor_visible(true);
                } else { event_loop.exit(); }
            }
            WindowEvent::KeyboardInput { event: KeyEvent {
                state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::Digit3), ..
            }, .. } => {
                state.probe_vis = !state.probe_vis;
                // Swap sprite texture.
                let raw: &[u8] = if state.probe_vis {
                    include_bytes!("../../probe.png")
                } else {
                    include_bytes!("../../spotlight.png")
                };
                let img = image::load_from_memory(raw)
                    .unwrap_or_else(|_| image::DynamicImage::new_rgba8(state.sprite_w, state.sprite_h))
                    .resize_exact(state.sprite_w, state.sprite_h, image::imageops::FilterType::Triangle)
                    .into_rgba8();
                if let Some(bb) = state.renderer.get_feature_mut::<BillboardsFeature>("billboards") {
                    bb.set_sprite(img.into_raw(), state.sprite_w, state.sprite_h);
                }
                // Swap billboard content in-place via persistent handles.
                // Remove old set then add new set, keeping handle vec updated.
                let new_data: &[BillboardInstance] = if state.probe_vis {
                    &state.probe_billboard_data
                } else {
                    &state.spotlight_billboards
                };
                // Remove all existing billboard handles.
                for id in state.billboard_ids.drain(..) {
                    state.renderer.remove_billboard(id);
                }
                // Register the new set.
                for inst in new_data {
                    state.billboard_ids.push(state.renderer.add_billboard(inst.clone()));
                }
            }
            WindowEvent::KeyboardInput { event: KeyEvent {
                state: ElementState::Pressed, physical_key: PhysicalKey::Code(KeyCode::Digit4), ..
            }, .. } => {
                let _ = state.renderer.start_live_portal_default();
            }
            WindowEvent::KeyboardInput { event: KeyEvent {
                state: ks, physical_key: PhysicalKey::Code(key), ..
            }, .. } => {
                match ks {
                    ElementState::Pressed  => { state.keys.insert(key); }
                    ElementState::Released => { state.keys.remove(&key); }
                }
            }
            WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => {
                if !state.cursor_grabbed {
                    let ok = state.window.set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked)).is_ok();
                    if ok { state.window.set_cursor_visible(false); state.cursor_grabbed = true; }
                }
            }
            WindowEvent::Resized(s) if s.width > 0 && s.height > 0 => {
                state.surface.configure(&state.device, &wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format: state.surface_format,
                    width: s.width, height: s.height, present_mode: wgpu::PresentMode::Fifo,
                    alpha_mode: wgpu::CompositeAlphaMode::Auto, view_formats: vec![],
                    desired_maximum_frame_latency: 2,
                });
                state.renderer.resize(s.width, s.height);
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
        if let Some(s) = &self.state { s.window.request_redraw(); }
    }
}

impl AppState {
    fn render(&mut self, dt: f32) {
        const SPEED: f32 = 4.0;
        const SENS:  f32 = 0.002;

        self.cam_yaw   += self.mouse_delta.0 * SENS;
        self.cam_pitch  = (self.cam_pitch - self.mouse_delta.1 * SENS).clamp(-1.4, 1.4);
        self.mouse_delta = (0.0, 0.0);

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let forward = glam::Vec3::new(sy * cp, sp, -cy * cp);
        let right   = glam::Vec3::new(cy, 0.0, sy);

        if self.keys.contains(&KeyCode::KeyW)      { self.cam_pos += forward * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyS)      { self.cam_pos -= forward * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyA)      { self.cam_pos -= right   * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyD)      { self.cam_pos += right   * SPEED * dt; }
        if self.keys.contains(&KeyCode::Space)     { self.cam_pos += glam::Vec3::Y * SPEED * dt; }
        if self.keys.contains(&KeyCode::ShiftLeft) { self.cam_pos -= glam::Vec3::Y * SPEED * dt; }

        let size   = self.window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;
        let time   = self.renderer.frame_count() as f32 * 0.016;

        let camera = Camera::perspective(
            self.cam_pos, self.cam_pos + forward, glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4, aspect, 0.1, 80.0, time,
        );

        let output = match self.surface.get_current_texture() {
            Ok(t) => t, Err(e) => { log::warn!("Surface: {:?}", e); return; }
        };
        let view = output.texture.create_view(&Default::default());

        // ── Zero per-frame allocations ────────────────────────────────────
        // Lights, billboards, and ambient were registered once in `resumed`.
        // Nothing to submit here — the renderer's persistent state is already
        // up to date.  Just render.
        if let Err(e) = self.renderer.render(&camera, &view, dt) {
            log::error!("Render: {:?}", e);
        }
        output.present();
    }
}
