//! Light-count benchmark — 150 simultaneous point lights with RC GI
//!
//! A large warehouse floor with a 6×6 pillar grid and scattered crates,
//! lit by a 10×15 array of 150 colored point lights at mid-height.
//! Tests deferred rendering + radiance cascades with high light count.
//!
//! Press 3 to toggle probe visualization, 4 for GPU timing.
//!
//! Controls:
//!   WASD        — move forward/left/back/right
//!   Space/Shift — move up/down
//!   Mouse drag  — look around (click to grab cursor)
//!   3           — toggle RC probe visualization ↔ light markers
//!   4           — toggle GPU timing printout (stderr)
//!   +/-         — increase/decrease light intensity
//!   Escape      — release cursor / exit



mod demo_portal;

use helio_render_v2::{Renderer, RendererConfig, Camera, GpuMesh, Scene, SceneLight};


use helio_render_v2::features::{
    FeatureRegistry, LightingFeature, BloomFeature, ShadowsFeature,
    BillboardsFeature, BillboardInstance, RadianceCascadesFeature,
};


use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId, CursorGrabMode},
};


use std::collections::HashSet;


use std::sync::Arc;

// ── Light grid ─────────────────────────────────────────────────────────────────
// 10 columns × 15 rows = 150 lights
const LIGHT_COLS:       usize = 10;
const LIGHT_ROWS:       usize = 15;
const LIGHT_HEIGHT:     f32   = 2.5;   // metres above floor
const LIGHT_SPACING_X:  f32   = 3.8;   // cols span ±18 m
const LIGHT_SPACING_Z:  f32   = 2.6;   // rows span ±19 m
const LIGHT_RANGE:      f32   = 7.0;
const LIGHT_INTENSITY:  f32   = 6.0;

/// Build the 150 lights deterministically from grid position.
fn build_lights() -> Vec<SceneLight> {
    let mut out = Vec::with_capacity(LIGHT_COLS * LIGHT_ROWS);
    let half_x = (LIGHT_COLS as f32 - 1.0) * 0.5 * LIGHT_SPACING_X;
    let half_z = (LIGHT_ROWS as f32 - 1.0) * 0.5 * LIGHT_SPACING_Z;
    for row in 0..LIGHT_ROWS {
        for col in 0..LIGHT_COLS {
            let x = col as f32 * LIGHT_SPACING_X - half_x;
            let z = row as f32 * LIGHT_SPACING_Z - half_z;
            // Spread hue smoothly across the grid for visual variety.
            let hue = (col * LIGHT_ROWS + row) as f32 / (LIGHT_COLS * LIGHT_ROWS) as f32;
            let color = hsv_to_rgb(hue, 0.75, 1.0);
            out.push(SceneLight::point(
                [x, LIGHT_HEIGHT, z],
                color,
                LIGHT_INTENSITY,
                LIGHT_RANGE,
            ));
        }
    }
    out
}

/// Simple HSV → linear-RGB conversion (no gamma).
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h6 = h * 6.0;
    let i  = h6.floor() as u32 % 6;
    let f  = h6 - h6.floor();
    let p  = v * (1.0 - s);
    let q  = v * (1.0 - s * f);
    let t  = v * (1.0 - s * (1.0 - f));
    match i {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

fn load_sprite() -> (Vec<u8>, u32, u32) {
    let img = image::load_from_memory(include_bytes!("../../spotlight.png"))
        .unwrap_or_else(|_| image::DynamicImage::new_rgba8(1, 1))
        .into_rgba8();
    let (w, h) = img.dimensions();
    (img.into_raw(), w, h)
}

const RC_WORLD_MIN: [f32; 3] = [-22.0, -0.5, -22.0];
const RC_WORLD_MAX: [f32; 3] = [ 22.0,  8.0,  22.0];

fn probe_billboards(world_min: [f32; 3], world_max: [f32; 3]) -> Vec<BillboardInstance> {
    use helio_render_v2::features::radiance_cascades::PROBE_DIMS;
    const COLORS: [[f32; 4]; 4] = [
        [0.0, 1.0, 1.0, 0.85],
        [0.0, 1.0, 0.0, 0.80],
        [1.0, 1.0, 0.0, 0.75],
        [1.0, 0.35, 0.0, 0.70],
    ];
    const SIZES: [[f32; 2]; 4] = [
        [0.035, 0.035],
        [0.075, 0.075],
        [0.140, 0.140],
        [0.260, 0.260],
    ];
    let mut out = Vec::new();
    for (c, &dim) in PROBE_DIMS.iter().enumerate() {
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    let [xmin, ymin, zmin] = world_min;
                    let [xmax, ymax, zmax] = world_max;
                    let t = 1.0 / (dim as f32 - 1.0).max(1.0);
                    let x = xmin + (xmax - xmin) * (i as f32 * t);
                    let y = ymin + (ymax - ymin) * (j as f32 * t);
                    let z = zmin + (zmax - zmin) * (k as f32 * t);
                    out.push(BillboardInstance::new([x, y, z], SIZES[c])
                        .with_color(COLORS[c]).with_screen_scale(true));
                }
            }
        }
    }
    out
}

// ── App / state ────────────────────────────────────────────────────────────────

fn main() {
    env_logger::init();
    log::info!("Starting Light Benchmark ({} lights)", LIGHT_COLS * LIGHT_ROWS);
    EventLoop::new()
        .expect("event loop")
        .run_app(&mut App::new())
        .expect("run");
}

struct App { state: Option<AppState> }
impl App { fn new() -> Self { Self { state: None } } }

struct AppState {
    window:         Arc<Window>,
    surface:        wgpu::Surface<'static>,
    device:         Arc<wgpu::Device>,
    surface_format: wgpu::TextureFormat,
    renderer:       Renderer,
    last_frame:     std::time::Instant,

    // geometry (built once, cloned into Scene each frame)
    floor:   GpuMesh,
    pillars: Vec<GpuMesh>,
    crates:  Vec<GpuMesh>,

    // camera
    cam_pos:        glam::Vec3,
    cam_yaw:        f32,
    cam_pitch:      f32,
    keys:           HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta:    (f32, f32),

    // debug
    probe_vis: bool,
    light_intensity_multiplier: f32,
    sprite_w: u32,
    sprite_h: u32,
    
    // event loop timing trackers
    time_render_end: Option<std::time::Instant>,
    time_about_to_wait_start: Option<std::time::Instant>,
    time_redraw_requested: Option<std::time::Instant>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }

        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_title(format!("Helio — Light Benchmark ({} lights)",
                        LIGHT_COLS * LIGHT_ROWS))
                    .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
            ).expect("window"),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(), ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })).expect("adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Benchmark Device"),
                required_features:
                    wgpu::Features::EXPERIMENTAL_RAY_QUERY |
                    wgpu::Features::TIMESTAMP_QUERY |
                    wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
                required_limits: wgpu::Limits::default()
                    .using_minimum_supported_acceleration_structure_values(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                trace: wgpu::Trace::Off,
            },
        )).expect("device");

        let device = Arc::new(device);
        let queue  = Arc::new(queue);
        let caps   = surface.get_capabilities(&adapter);
        let fmt    = caps.formats.iter().find(|f| f.is_srgb()).copied()
                         .unwrap_or(caps.formats[0]);
        let size   = window.inner_size();
        surface.configure(&device, &wgpu::SurfaceConfiguration {
            usage:    wgpu::TextureUsages::RENDER_ATTACHMENT,
            format:   fmt,
            width:    size.width,
            height:   size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode:   caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 1,
        });

        let (sprite_rgba, sprite_w, sprite_h) = load_sprite();
        let features = FeatureRegistry::builder()
            .with_feature(LightingFeature::new())
            .with_feature(ShadowsFeature::new().with_max_lights(16))
            .with_feature(BloomFeature::new().with_intensity(0.5).with_threshold(1.0))
            .with_feature(
                BillboardsFeature::new()
                    .with_sprite(sprite_rgba, sprite_w, sprite_h)
                    .with_max_instances(5000),
            )
            .with_feature(
                RadianceCascadesFeature::new()
                    .with_world_bounds(RC_WORLD_MIN, RC_WORLD_MAX)
            )
            .build();

        let mut renderer = Renderer::new(device.clone(), queue.clone(), RendererConfig::new(
            size.width, size.height, fmt, features
        )).expect("renderer");

        // ── Geometry ──────────────────────────────────────────────────────────
        let floor = GpuMesh::plane(&device, [0.0, 0.0, 0.0], 20.0);

        // 6 × 6 = 36 pillars, spaced 4 m apart (–10 … +10 m on each axis)
        let mut pillars = Vec::new();
        for ix in -2..=3i32 {
            for iz in -2..=3i32 {
                pillars.push(GpuMesh::cube(
                    &device,
                    [ix as f32 * 4.0, 0.5, iz as f32 * 4.0],
                    0.4,
                ));
            }
        }

        // Scattered crates as props
        let crate_defs: &[([f32; 3], f32)] = &[
            ([-8.0,  0.3,  -6.5], 0.30), ([  8.5, 0.3,   4.0], 0.30),
            ([-5.0,  0.3,  10.5], 0.25), ([  7.0, 0.3,  -9.5], 0.30),
            ([-13.0, 0.3,   2.0], 0.35), ([ 13.0, 0.3,  -3.0], 0.35),
            ([-10.0, 0.3, -13.0], 0.30), ([ 10.0, 0.3,  13.0], 0.30),
            ([  3.0, 0.3, -15.5], 0.25), ([ -3.0, 0.3,  15.5], 0.25),
            ([  0.0, 0.3,   8.0], 0.30), ([ -1.0, 0.3,  -8.0], 0.28),
            ([ 15.0, 0.3,   7.0], 0.30), ([-15.0, 0.3,  -7.0], 0.30),
            ([  6.0, 0.3,  17.0], 0.25), ([ -6.0, 0.3, -17.0], 0.25),
        ];
        let crates = crate_defs.iter()
            .map(|&(pos, hs)| GpuMesh::cube(&device, pos, hs))
            .collect();
        demo_portal::enable_live_dashboard(&mut renderer);

        self.state = Some(AppState {
            window, surface, device, surface_format: fmt, renderer,
            last_frame: std::time::Instant::now(),
            floor, pillars, crates,
            cam_pos:        glam::Vec3::new(0.0, 4.0, 22.0),
            cam_yaw:        0.0,
            cam_pitch:      -0.18,
            keys:           HashSet::new(),
            cursor_grabbed: false,
            mouse_delta:    (0.0, 0.0),
            probe_vis: false,
            light_intensity_multiplier: 1.0,
            sprite_w, sprite_h,
            time_render_end: None,
            time_about_to_wait_start: None,
            time_redraw_requested: None,
        });
    }

    fn window_event(
        &mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => { event_loop.exit(); }

            WindowEvent::KeyboardInput { event: KeyEvent {
                state: ElementState::Pressed,
                physical_key: PhysicalKey::Code(KeyCode::Escape), ..
            }, .. } => {
                if state.cursor_grabbed {
                    state.cursor_grabbed = false;
                    let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                    state.window.set_cursor_visible(true);
                } else { event_loop.exit(); }
            }

            // Toggle probe visualization
            WindowEvent::KeyboardInput { event: KeyEvent {
                state: ElementState::Pressed,
                physical_key: PhysicalKey::Code(KeyCode::Digit3), ..
            }, .. } => {
                state.probe_vis = !state.probe_vis;
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
            }

            // Toggle GPU timing printout
            WindowEvent::KeyboardInput { event: KeyEvent {
                state: ElementState::Pressed,
                physical_key: PhysicalKey::Code(KeyCode::Digit4), ..
            }, .. } => { state.renderer.debug_key_pressed(); }

            // Decrease light intensity
            WindowEvent::KeyboardInput { event: KeyEvent {
                state: ElementState::Pressed,
                physical_key: PhysicalKey::Code(KeyCode::Minus), ..
            }, .. } => {
                state.light_intensity_multiplier = (state.light_intensity_multiplier - 0.1).max(0.1);
                eprintln!("Light intensity: {:.1}x", state.light_intensity_multiplier);
            }

            // Increase light intensity
            WindowEvent::KeyboardInput { event: KeyEvent {
                state: ElementState::Pressed,
                physical_key: PhysicalKey::Code(KeyCode::Equal), ..
            }, .. } => {
                state.light_intensity_multiplier = (state.light_intensity_multiplier + 0.1).min(5.0);
                eprintln!("Light intensity: {:.1}x", state.light_intensity_multiplier);
            }

            WindowEvent::KeyboardInput { event: KeyEvent {
                state: ks, physical_key: PhysicalKey::Code(key), ..
            }, .. } => {
                match ks {
                    ElementState::Pressed  => { state.keys.insert(key); }
                    ElementState::Released => { state.keys.remove(&key); }
                }
            }

            WindowEvent::MouseInput {
                state: ElementState::Pressed, button: MouseButton::Left, ..
            } => {
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
                state.surface.configure(&state.device, &wgpu::SurfaceConfiguration {
                    usage:    wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format:   state.surface_format,
                    width:    s.width,
                    height:   s.height,
                    present_mode: wgpu::PresentMode::Fifo,
                    alpha_mode:   wgpu::CompositeAlphaMode::Auto,
                    view_formats: vec![],
                    desired_maximum_frame_latency: 1,
                });
                state.renderer.resize(s.width, s.height);
            }

            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                
                // Track FULL cycle from last render_end to this RedrawRequested
                if let Some(last_render_end) = state.time_render_end {
                    let full_cycle_ms = last_render_end.elapsed().as_secs_f32() * 1000.0;
                    if state.renderer.frame_count() % 60 == 0 {
                        eprintln!("🔄 render_end → next RedrawRequested: {:.2}ms", full_cycle_ms);
                    }
                }
                
                // Track time from about_to_wait to RedrawRequested
                if let Some(about_to_wait_start) = state.time_about_to_wait_start {
                    let gap_ms = about_to_wait_start.elapsed().as_secs_f32() * 1000.0;
                    if gap_ms > 2.0 {
                        eprintln!("⏱️  about_to_wait → RedrawRequested: {:.2}ms", gap_ms);
                    }
                }
                
                state.time_redraw_requested = Some(now);
                let dt  = (now - state.last_frame).as_secs_f32();
                state.last_frame = now;
                state.render(dt);
                // Don't call request_redraw() here - about_to_wait() handles it
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self, _: &ActiveEventLoop, _: winit::event::DeviceId, event: DeviceEvent,
    ) {
        let Some(state) = &mut self.state else { return };
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if state.cursor_grabbed {
                state.mouse_delta.0 += dx as f32;
                state.mouse_delta.1 += dy as f32;
            }
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(s) = &mut self.state {
            let now = std::time::Instant::now();
            if let Some(render_end) = s.time_render_end {
                let gap_ms = render_end.elapsed().as_secs_f32() * 1000.0;
                if gap_ms > 2.0 {
                    eprintln!("⏱️  render_end → about_to_wait: {:.2}ms", gap_ms);
                }
            }
            s.time_about_to_wait_start = Some(now);
            s.window.request_redraw();
        }
    }
}

impl AppState {
    fn render(&mut self, dt: f32) {
        // Track time from RedrawRequested event to start of render()
        if let Some(redraw_time) = self.time_redraw_requested {
            let gap_ms = redraw_time.elapsed().as_secs_f32() * 1000.0;
            if gap_ms > 2.0 {
                eprintln!("⏱️  RedrawRequested → render(): {:.2}ms", gap_ms);
            }
        }
        
        const SPEED: f32 = 8.0;
        const SENS:  f32 = 0.002;

        self.cam_yaw   += self.mouse_delta.0 * SENS;
        self.cam_pitch  = (self.cam_pitch - self.mouse_delta.1 * SENS).clamp(-1.5, 1.5);
        self.mouse_delta = (0.0, 0.0);

        let (sy, cy) = self.cam_yaw.sin_cos();
        let (sp, cp) = self.cam_pitch.sin_cos();
        let forward  = glam::Vec3::new(sy * cp, sp, -cy * cp);
        let right    = glam::Vec3::new(cy, 0.0, sy);

        if self.keys.contains(&KeyCode::KeyW)      { self.cam_pos += forward * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyS)      { self.cam_pos -= forward * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyA)      { self.cam_pos -= right   * SPEED * dt; }
        if self.keys.contains(&KeyCode::KeyD)      { self.cam_pos += right   * SPEED * dt; }
        if self.keys.contains(&KeyCode::Space)     { self.cam_pos.y += SPEED * dt; }
        if self.keys.contains(&KeyCode::ShiftLeft) { self.cam_pos.y -= SPEED * dt; }

        let size   = self.window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;
        let time   = self.renderer.frame_count() as f32 * 0.016;

        let camera = Camera::perspective(
            self.cam_pos,
            self.cam_pos + forward,
            glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            aspect,
            0.1,
            300.0,
            time,
        );

        // Time get_current_texture() - this can block waiting for GPU
        let get_texture_start = std::time::Instant::now();
        let output = match self.surface.get_current_texture() {
            Ok(t)  => t,
            Err(e) => { log::warn!("Surface error: {:?}", e); return; }
        };
        let get_texture_ms = get_texture_start.elapsed().as_secs_f32() * 1000.0;
        if get_texture_ms > 10.0 {
            eprintln!("⚠️  get_current_texture() blocked for {:.2}ms", get_texture_ms);
        }
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Time scene construction
        let scene_build_start = std::time::Instant::now();
        
        let mut lights = build_lights();
        
        // Smooth fade-in over first 2 seconds (~120 frames at 60fps)
        let fade_in_frames = 120.0;
        let frame_age = (self.renderer.frame_count() as f32).min(fade_in_frames);
        let time_fade = if frame_age < fade_in_frames {
            // Smoothstep: smooth curve from 0 to 1
            let t = frame_age / fade_in_frames;
            t * t * (3.0 - 2.0 * t)  // Hermite smoothstep
        } else {
            1.0
        };
        
        // Apply time-based fade-in only (GPU will handle distance fade)
        for light in &mut lights {
            light.intensity *= self.light_intensity_multiplier * time_fade;
        }

        let mut scene = Scene::new();
        scene.ambient_color     = [0.03, 0.03, 0.04];
        scene.ambient_intensity = 1.0;
        scene = scene.add_object(self.floor.clone());
        for p in &self.pillars { scene = scene.add_object(p.clone()); }
        for c in &self.crates  { scene = scene.add_object(c.clone()); }
        for l in &lights       { scene = scene.add_light(l.clone()); }

        if self.probe_vis {
            for b in probe_billboards(RC_WORLD_MIN, RC_WORLD_MAX) {
                scene = scene.add_billboard(b);
            }
        } else {
            for l in &lights {
                let col = l.color;
                scene = scene.add_billboard(
                    BillboardInstance::new(l.position, [0.15, 0.15])
                        .with_color([col[0], col[1], col[2], 0.85])
                        .with_screen_scale(true),
                );
            }
        }
        
        let scene_build_ms = scene_build_start.elapsed().as_secs_f32() * 1000.0;
        if scene_build_ms > 10.0 {
            eprintln!("⚠️  Scene construction took {:.2}ms", scene_build_ms);
        }

        if let Err(e) = self.renderer.render_scene(&scene, &camera, &view, dt) {
            log::error!("Render error: {:?}", e);
        }
        
        // Time the present() call to see if it's blocking
        let present_start = std::time::Instant::now();
        output.present();
        let present_ms = present_start.elapsed().as_secs_f32() * 1000.0;
        
        // Track when render() completes
        let render_complete = std::time::Instant::now();
        self.time_render_end = Some(render_complete);
        
        // Print full render cycle timing every 60 frames
        if self.renderer.frame_count() % 60 == 0 {
            let total_render_ms = if let Some(redraw_time) = self.time_redraw_requested {
                redraw_time.elapsed().as_secs_f32() * 1000.0
            } else {
                0.0
            };
            eprintln!("🔄 Full cycle: total_render={:.2}ms, present={:.2}ms", total_render_ms, present_ms);
        }
    }
}
