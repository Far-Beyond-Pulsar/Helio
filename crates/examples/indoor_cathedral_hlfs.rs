//! Indoor cathedral example with HLFS ScreenSpace visibility
//!
//! A Gothic interior with ribbed vaults, clustered limestone piers, marble
//! paving, carved oak pews, bronze chandeliers and leaded stained glass.
//! Panes use alpha blending, with explicit thin-sheet RGB shadow transmission
//! in RT mode. Refraction and caustics are not simulated.
//! RT defaults to one daylight sun plus interior lights. Set
//! `HLFS_LEGACY_CATHEDRAL_LIGHTS=1` for the multi-window transmission stress setup.
//!
//! HLFS uses hierarchical light culling, visibility-guided sampling and
//! temporal/spatial filtering with a bounded shadow budget per shading pixel.
//! `--capture <directory>` renders a deterministic offscreen camera path.
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Mouse drag  — look around (click to grab cursor)
//!   Escape      — release cursor / exit

mod hlfs_capture;
mod architectural_mesh;
mod cathedral_detail;
mod cathedral_large;
mod v3_demo_common;

use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera,
    HelioAction, HelioCommandBridge, Renderer, RendererBuilder, RendererConfig,
};
use helio_default_graphs::build_hlfs_graph_with_context;
use helio_pass_perf_overlay::PerfOverlayMode;
use pulsar_scenedb::{Entity, SceneDb, World};
use v3_demo_common::{
    new_scene_db_with_gpu_mirror, point_light,
    scene_db_handle, spawn_indoor_cathedral_sky, spawn_light,
    update_light,
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
    (-10.3, 9.0, -22.0, 0.8, 0.2, 1.0), // violet
    (-10.3, 9.0, -6.0, 0.2, 0.7, 1.0),  // sky blue
    (-10.3, 9.0, 10.0, 0.2, 1.0, 0.4),   // emerald
    (-10.3, 9.0, 18.0, 1.0, 0.7, 0.1),  // gold
    // Right wall (x ≈ +10.5)
    (10.3, 9.0, -22.0, 1.0, 0.2, 0.3), // ruby
    (10.3, 9.0, -6.0, 1.0, 0.5, 0.1),  // amber
    (10.3, 9.0, 10.0, 0.1, 0.8, 0.9),   // teal
    (10.3, 9.0, 18.0, 0.9, 0.1, 0.7),  // magenta
    // Rose window above entrance (back wall, z ≈ +28)
    (0.0, 13.0, 27.0, 1.0, 0.75, 0.3), // warm gold
];

// Chandelier positions (x=0, hanging from y≈19.5, at z intervals)
const CHANDELIER_Z: &[f32] = &[-16.0, 0.0, 16.0];
const LARGE_CHANDELIER_Z: &[f32] = &[-54.0, -36.0, -18.0, 0.0, 18.0, 36.0, 54.0];

// Candle cluster positions near the altar (z ≈ -24)
const CANDLES: &[(f32, f32, f32)] = &[
    (-3.0, 1.6, -23.5),
    (-1.5, 1.6, -23.0),
    (0.0, 1.6, -23.5),
    (1.5, 1.6, -23.0),
    (3.0, 1.6, -23.5),
];
const LARGE_CANDLES: &[(f32, f32, f32)] = &[
    (-4.0, 1.6, -64.0), (-2.0, 1.6, -63.5), (0.0, 1.6, -64.0),
    (2.0, 1.6, -63.5), (4.0, 1.6, -64.0),
];

fn main() {
    env_logger::init();
    if let Some(directory) = std::env::args()
        .nth(1)
        .filter(|a| a == "--capture")
        .and_then(|_| std::env::args().nth(2))
    {
        hlfs_capture::run(&directory, populate_cathedral);
        return;
    }
    if let Some(directory) = std::env::args()
        .nth(1)
        .filter(|a| a == "--capture-large")
        .and_then(|_| std::env::args().nth(2))
    {
        hlfs_capture::run_scene(&directory, "cathedral_large", populate_large_cathedral,
            large_cathedral_camera);
        return;
    }
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
    queue: Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
    renderer: Arc<Mutex<Renderer>>,
    action_rx: Receiver<HelioAction>,
    last_frame: std::time::Instant,

    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    alt_pressed: bool,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    // Debug
    debug_mode: u32,
    perf_overlay_mode: PerfOverlayMode,
    debug_overlay_enabled: bool,

    scene_db: SceneDb,
    acceleration: Option<helio_pass_hlfs::SceneDbRayTracing>,

    // Scene state
    chandelier_light_ids: Vec<Entity>,
    candle_light_ids: Vec<Entity>,
    large: bool,
    start_time: std::time::Instant,
    motion_frame: u32,
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
                        .with_title("Helio – Indoor Cathedral (HLFS)")
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
        eprintln!("Interactive cathedral adapter: {:?}", adapter.get_info());
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

        let config = RendererConfig::new(size.width, size.height, format)
            .with_tsr_quality(helio_pass_tsr::TsrQuality::Native)
            .with_shadow_quality(helio::ShadowQuality::High)
            .with_ssr(true)
            .with_environment_reflections(true);
        let mut scene_db = new_scene_db_with_gpu_mirror(&device, &queue);
        let large = std::env::var_os("HLFS_LARGE_CATHEDRAL").is_some();
        let (chandelier_light_ids, candle_light_ids) = if large {
            populate_large_cathedral(&mut scene_db.world)
        } else {
            populate_cathedral(&mut scene_db.world)
        };
        let ray_traced = std::env::var_os("HLFS_RT").is_some();
        eprintln!("Interactive cathedral: large={large} ray_traced={ray_traced} presampled={}",
            std::env::var_os("HLFS_PRESAMPLED").is_some());
        if ray_traced {
            // The renderer captures SceneDB's initial light flags at build.
            // Match the offscreen path by setting RT flags before building it.
            hlfs_capture::enable_ray_shadows(&mut scene_db.world);
        }

        let mut scene_handle = scene_db_handle(&scene_db);
        let stone_store = hlfs_capture::architectural_materials::load(&device, &queue, &mut scene_db.world);
        let has_stone = stone_store.is_some();
        if let Some(store) = stone_store { scene_handle = scene_handle.with_texture_store(store).unwrap(); }
        let mut renderer = RendererBuilder::new(config, scene_handle)
            .with_external_device()
            .with_editor_mode(false)
            .with_pass_build_context(Box::new(build_hlfs_graph_with_context))
            .build(device.clone(), queue.clone(), size.width, size.height, format);
        if has_stone { hlfs_capture::architectural_materials::configure_sampler(&mut renderer); }
        let mut acceleration = if ray_traced {
            let config = if std::env::var_os("HLFS_PRESAMPLED").is_some() {
                helio_pass_hlfs::HlfsConfig::ray_traced_presampled()
            } else {
                helio_pass_hlfs::HlfsConfig { mode: helio_pass_hlfs::HlfsMode::RayTraced, ..Default::default() }
            };
            renderer.set_graph_rebuild_hook(move |graph, device| {
                graph.find_pass_mut::<helio_pass_hlfs::HlfsPass>().expect("HLFS pass")
                    .set_config(device, config);
            });
            eprintln!("Interactive HLFS configuration: {:?}",
                renderer.find_pass_mut::<helio_pass_hlfs::HlfsPass>().unwrap().config());
            Some(helio_pass_hlfs::SceneDbRayTracing::new(device.clone(), queue.clone()))
        } else { None };
        renderer.set_ambient([0.10, 0.09, 0.085], 1.0);
        renderer.set_clear_color([0.0, 0.0, 0.0, 1.0]);

        let warmup_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Cathedral startup warmup"),
            size: wgpu::Extent3d { width: size.width, height: size.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let warmup_view = warmup_texture.create_view(&Default::default());
        let aspect = size.width as f32 / size.height.max(1) as f32;
        let warmup_camera = if large { large_cathedral_camera(0.0, aspect) } else {
            let pos = glam::Vec3::new(0.0, 2.0, 24.0);
            Camera::perspective_look_at(pos,
                pos + glam::Vec3::new(0.0, 0.065_f32.sin(), -0.065_f32.cos()),
                glam::Vec3::Y, std::f32::consts::FRAC_PI_4, aspect, 0.1, 200.0)
        };
        if let Some(acceleration) = acceleration.as_mut() {
            v3_demo_common::flush_scene_db(&scene_db, &queue);
            acceleration.prepare(&scene_db.world).expect("cathedral RT geometry");
        }
        hlfs_capture::warm_up_cathedral(&scene_db, &mut renderer, acceleration.as_ref(),
            &device, &queue, &warmup_camera, &warmup_view);

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
            queue,
            surface_format: format,
            renderer,
            action_rx,
            last_frame: std::time::Instant::now(),
            scene_db,
            acceleration,
            // Start at entrance, looking toward the altar
            cam_pos: if large { glam::Vec3::new(0.0, 2.3, 67.0) }
                else { glam::Vec3::new(0.0, 2.0, 24.0) },
            cam_yaw: 0.0,
            cam_pitch: 0.065,
            keys: HashSet::new(),
            alt_pressed: false,
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            debug_mode: 0,
            perf_overlay_mode: PerfOverlayMode::Disabled,
            debug_overlay_enabled: false,
            chandelier_light_ids,
            candle_light_ids,
            large,
            start_time: std::time::Instant::now(),
            motion_frame: 0,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Focused(false) => {
                // Some platforms swallow key-up while a system menu or another
                // window owns focus. Never keep flying on a stale movement key.
                state.keys.clear();
                state.alt_pressed = false;
                state.mouse_delta = (0.0, 0.0);
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                state.alt_pressed = modifiers.state().alt_key();
                if state.alt_pressed {
                    state.keys.clear();
                }
            }
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

            // F2: cycle perf overlay modes
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
                    if let Some(pass) =
                        renderer.find_pass_mut::<helio_pass_perf_overlay::PerfOverlayPass>()
                    {
                        pass.set_mode(state.perf_overlay_mode);
                    }
                }
                println!("[debug] perf overlay mode = {:?}", state.perf_overlay_mode);
            }

            // F3: toggle debug overlay
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
                    if let Some(pass) =
                        renderer.find_pass_mut::<helio_pass_debug_overlay::DebugOverlayPass>()
                    {
                        pass.set_enabled(state.debug_overlay_enabled);
                    }
                }
                println!("[debug] debug overlay = {:?}", state.debug_overlay_enabled);
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
                    if !state.alt_pressed {
                        state.keys.insert(key);
                    }
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

        let mut camera = Camera::perspective_look_at(
            self.cam_pos,
            self.cam_pos + forward,
            glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            aspect,
            0.1,
            200.0,
        );
        configure_cathedral_fog(&mut camera, self.large);
        if self.large && std::env::var_os("HLFS_LIVE_MOTION_TEST").is_some() {
            // Travel the same path as --capture-large, then reverse smoothly.
            // This exercises the real swapchain and resize path while moving.
            let phase = (self.motion_frame % 600) as f32 / 300.0;
            let t = if phase <= 1.0 { phase } else { 2.0 - phase };
            camera = large_cathedral_camera(t, aspect);
            self.motion_frame = self.motion_frame.wrapping_add(1);
        }

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

        let with_shadows = |mut light: helio::GpuLight| {
            light.set_ray_traced_shadows(self.acceleration.is_some());
            light
        };
        // Update flickering chandelier intensities
        let chandelier_z = if self.large { LARGE_CHANDELIER_Z } else { CHANDELIER_Z };
        let candles = if self.large { LARGE_CANDLES } else { CANDLES };
        for (i, &id) in self.chandelier_light_ids.iter().enumerate() {
            let z = chandelier_z[i];
            update_light(
                &mut self.scene_db.world,
                id,
                with_shadows(point_light([0.0_f32, if self.large { 31.0 } else { 15.0 }, z],
                    [1.0, 0.92, 0.78], 160.0 * flicker, 22.0)),
            );
        }
        // Update flickering candle intensities
        for (i, &id) in self.candle_light_ids.iter().enumerate() {
            let (x, y, z) = candles[i];
            update_light(
                &mut self.scene_db.world,
                id,
                with_shadows(point_light([x, y, z], [1.0, 0.6, 0.15], 8.0 * cflicker, 4.0)),
            );
        }

        // Scene state is persistent — no per-frame setup needed.

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            _ => return,
        };
        let view = output.texture.create_view(&Default::default());

        v3_demo_common::flush_scene_db(&self.scene_db, &self.queue);
        if let Some(acceleration) = &self.acceleration {
            renderer.set_ray_tracing_frame_with_transmission(acceleration.tlas(), acceleration.transmission());
        }
        if let Err(e) = renderer.render(&camera, &view) {
            log::error!("Render: {:?}", e);
        }
        if self.acceleration.is_some() {
            static CONFIG_LOGGED: std::sync::Once = std::sync::Once::new();
            CONFIG_LOGGED.call_once(|| eprintln!("HLFS after first live render: {:?}",
                renderer.find_pass_mut::<helio_pass_hlfs::HlfsPass>().unwrap().config()));
        }
        self.queue.present(output);
    }
}

fn populate_cathedral(world: &mut World) -> (Vec<Entity>, Vec<Entity>) {
    spawn_indoor_cathedral_sky(world);
    cathedral_detail::populate(world);
    populate_cathedral_lights(world, false)
}

fn configure_cathedral_fog(camera: &mut Camera, large: bool) {
    if !large || std::env::var_os("HLFS_NO_FOG").is_some() { return; }
    let settings = &mut camera.postprocess_settings;
    settings.fog_enabled = true;
    settings.fog_density = 0.004;
    settings.fog_color = [0.57, 0.55, 0.51];
    settings.fog_scattering_anisotropy = 0.65;
    settings.fog_max_distance = 180.0;
    settings.fog_start_distance = 1.5;
}

fn large_cathedral_camera(t: f32, aspect: f32) -> Camera {
    let mut camera = Camera::perspective_look_at(
        glam::Vec3::new(4.0 * t, 2.3, 67.0 - 29.0 * t),
        glam::Vec3::new(0.0, 10.0, -68.0), glam::Vec3::Y,
        std::f32::consts::FRAC_PI_4, aspect, 0.1, 200.0,
    );
    configure_cathedral_fog(&mut camera, true);
    camera
}

fn populate_large_cathedral(world: &mut World) -> (Vec<Entity>, Vec<Entity>) {
    spawn_indoor_cathedral_sky(world);
    cathedral_large::populate(world);
    populate_cathedral_lights(world, true)
}

fn populate_cathedral_lights(world: &mut World, large: bool) -> (Vec<Entity>, Vec<Entity>) {
    let chandelier_z = if large { LARGE_CHANDELIER_Z } else { CHANDELIER_Z };
    let candles = if large { LARGE_CANDLES } else { CANDLES };

    // Register lights (chandelier & candle light_ids stored for per-frame flicker updates)
    let mut chandelier_light_ids = Vec::new();
    for &z in chandelier_z {
        chandelier_light_ids.push(spawn_light(
            world,
            point_light([0.0_f32, if large { 31.0 } else { 15.0 }, z],
                [1.0, 0.92, 0.78], 160.0, 22.0),
        ));
    }
    // A single exterior sun supplies a coherent daylight direction. Keep the
    // older multi-window emitter setup as an explicit transmission stress case.
    if std::env::var_os("HLFS_RT").is_some()
        && std::env::var_os("HLFS_LEGACY_CATHEDRAL_LIGHTS").is_none()
    {
        spawn_light(world, v3_demo_common::directional_light(
            [0.80, -0.48, -0.30], [1.0, 0.94, 0.84], 4.0,
        ));
    } else {
        // Stained glass shafts — static, no need to store ids
        // RT uses white exterior sources: pane materials supply the transmitted tint.
        for &(x, y, z, r, g, b) in GLASS_LIGHTS {
            let (x, y, z) = if large { (x.signum() * 21.5, y * 1.8, z * 2.4) }
                else { (x, y, z) };
            let light = if std::env::var_os("HLFS_RT").is_some() {
                let position = if x == 0.0 { [0.0, if large { 34.0 } else { 17.0 },
                    if large { 78.0 } else { 34.0 }] }
                    else { [x.signum() * if large { 29.0 } else { 16.0 },
                        if large { 24.0 } else { 12.0 }, z] };
                point_light(position, [1.0; 3], 2500.0, 65.0)
            } else {
                point_light([x, y, z], [r, g, b], 35.0, 10.0)
            };
            spawn_light(world, light);
        }
    }
    let mut candle_light_ids = Vec::new();
    for &(x, y, z) in candles {
        candle_light_ids.push(spawn_light(
            world,
            point_light([x, y, z], [1.0, 0.6, 0.15], 8.0, 4.0),
        ));
    }

    (chandelier_light_ids, candle_light_ids)
}
