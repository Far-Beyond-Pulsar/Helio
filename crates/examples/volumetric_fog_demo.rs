//! Volumetric fog + light shafts test scene.
//!
//! A roofed colonnade: two rows of pillars with gaps between them, and a low sun
//! raking through those gaps. The gaps are the point — shafts are only legible
//! where a shadow caster chops the light into slices, so a low angle through a
//! gapped wall shows far more than an open field would.
//!
//! What it exercises:
//!   - Uniform and height-based fog (M)
//!   - Light shafts through the CSM shadow atlas (G) — shafts must line up with
//!     the pillar shadows on the floor; if they don't, the fog pass and deferred
//!     lighting disagree about cascade selection.
//!   - Henyey-Greenstein anisotropy (3/4) — face the sun and the halo should
//!     brighten at positive g, flatten at 0.
//!   - Post-process volume fog blending: the camera carries thin haze, and a
//!     dense pocket sits mid-hall between z = -6 and z = 6. Walk in and out.
//!
//! Controls:
//!   WASD        — move, Space/Shift up/down, mouse drag to look (click to grab)
//!   Q/E         — rotate sun (low angles make the best shafts)
//!   F           — fog on/off
//!   G           — light shafts on/off
//!   M           — uniform / height-based fog
//!   1 / 2       — fog density down / up
//!   3 / 4       — anisotropy (g) down / up
//!   Escape      — release cursor / exit

mod v3_demo_common;

use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera,
    DebugDrawState, LightId, Renderer, RendererConfig, Scene,
};
use helio_default_graphs::build_default_graph;
use libhelio::{FogMode, PostProcessSettings, PostProcessVolumeDescriptor};
use v3_demo_common::{box_mesh, directional_light, make_material, plane_mesh};

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use std::collections::HashSet;
use std::sync::Arc;

// Hall dimensions. Pillars run along z at +/-HALL_HALF_X, with GAP between them.
const HALL_HALF_X: f32 = 7.0;
const HALL_HALF_Z: f32 = 22.0;
const ROOF_Y: f32 = 9.0;
const PILLAR_SPACING: f32 = 4.0;
const PILLAR_HALF_W: f32 = 0.7;

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
    surface_format: wgpu::TextureFormat,
    renderer: Renderer,
    last_frame: std::time::Instant,

    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    keys: HashSet<KeyCode>,
    just_pressed: Vec<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),

    sun_angle: f32,
    sun_light_id: LightId,

    // Fog state, pushed onto the camera every frame.
    fog_enabled: bool,
    shafts_enabled: bool,
    fog_mode: FogMode,
    fog_density: f32,
    fog_anisotropy: f32,
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
                        .with_title("Helio – Volumetric Fog & Light Shafts")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .expect("Failed to create window"),
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: true,
        }))
        .expect("Failed to find adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Main Device"),
            required_features: required_wgpu_features(adapter.features()),
            required_limits: required_wgpu_limits(adapter.limits()),
            experimental_features: required_experimental_features(adapter.features()),
            ..Default::default()
        }))
        .expect("Failed to create device");

        device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| {
            panic!("[GPU UNCAPTURED ERROR] {:?}", e);
        }));
        let info = adapter.get_info();
        println!(
            "[WGPU] Backend: {:?}, Device: {}, Driver: {}",
            info.backend, info.name, info.driver
        );
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let size = window.inner_size();
        let cfg = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&device, &cfg);

        // Full internal resolution. RendererConfig defaults to render_scale 0.75,
        // which upscales 960x540 -> 1280x720 and shows as soft, stair-stepped
        // edges — fine in a game demo, but here it would be mistaken for fog
        // quality. Fog accumulates at a quarter of *this*, so the base wants to
        // be honest.
        let config = RendererConfig::new(size.width, size.height, surface_format)
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
        let graph = build_default_graph(
            &device,
            &queue,
            &scene,
            config,
            debug_state.clone(),
            &debug_camera_buf,
            &cull_stats_buf,
            None,
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

        let stone = renderer.scene_mut().insert_material(make_material(
            [0.62, 0.60, 0.58, 1.0],
            0.85,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
        ));

        // Floor
        let floor = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(plane_mesh([0.0, 0.0, 0.0], 40.0)))
            .as_mesh()
            .unwrap();
        let _ = v3_demo_common::insert_object(&mut renderer, floor, stone, glam::Mat4::IDENTITY, 40.0);

        // Roof — without it the sun lights everything and there is nothing to
        // slice the light into shafts.
        let roof = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(box_mesh(
                [0.0, 0.0, 0.0],
                [HALL_HALF_X + 1.0, 0.3, HALL_HALF_Z],
            )))
            .as_mesh()
            .unwrap();
        let _ = v3_demo_common::insert_object(
            &mut renderer,
            roof,
            stone,
            glam::Mat4::from_translation(glam::Vec3::new(0.0, ROOF_Y, 0.0)),
            HALL_HALF_Z,
        );

        // Two rows of pillars. The gaps between them are what the sun cuts through.
        let pillar = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::mesh(box_mesh(
                [0.0, 0.0, 0.0],
                [PILLAR_HALF_W, ROOF_Y * 0.5, PILLAR_HALF_W],
            )))
            .as_mesh()
            .unwrap();

        let count = (HALL_HALF_Z * 2.0 / PILLAR_SPACING) as i32;
        for i in 0..=count {
            let z = -HALL_HALF_Z + i as f32 * PILLAR_SPACING;
            for side in [-1.0_f32, 1.0] {
                let _ = v3_demo_common::insert_object(
                    &mut renderer,
                    pillar,
                    stone,
                    glam::Mat4::from_translation(glam::Vec3::new(
                        side * HALL_HALF_X,
                        ROOF_Y * 0.5,
                        z,
                    )),
                    ROOF_Y * 0.5,
                );
            }
        }

        // Sun. god_rays_enabled is what opts a light into fog in-scattering —
        // lights without it still light surfaces but cost the fog pass nothing.
        let mut sun = directional_light(sun_light_dir(1.0), [1.0, 0.9, 0.75], 4.0);
        sun.god_rays_enabled = 1;
        let sun_light_id = renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::light(sun))
            .as_light()
            .unwrap();

        renderer.set_ambient([0.10, 0.12, 0.18], 0.05);

        // A denser pocket of fog mid-hall.
        //
        // The camera carries thin haze and this only raises the density, rather
        // than switching fog on from off. That is deliberate: the blender applies
        // bool fields via `select(base, vol, t > 0.5)`, and a lone volume at
        // blend_weight 1.0 lands on exactly t = 0.5, so it cannot flip an enable
        // flag. Floats blend fine, which is what this volume varies.
        renderer
            .scene_mut()
            .insert_actor(helio::SceneActor::post_process_volume(
                PostProcessVolumeDescriptor {
                    bounds_min: [-HALL_HALF_X, 0.0, -6.0],
                    bounds_max: [HALL_HALF_X, ROOF_Y, 6.0],
                    priority: 10.0,
                    blend_radius: 4.0,
                    blend_weight: 1.0,
                    unbound: false,
                    settings: PostProcessSettings {
                        fog_enabled: true,
                        fog_density: 0.08,
                        fog_color: [0.75, 0.80, 0.95],
                        fog_scattering_anisotropy: 0.7,
                        ..Default::default()
                    },
                },
            ));

        print_help();

        self.state = Some(AppState {
            window,
            surface,
            device,
            surface_format,
            renderer,
            last_frame: std::time::Instant::now(),
            cam_pos: glam::Vec3::new(0.0, 2.0, 16.0),
            cam_yaw: 0.0,
            cam_pitch: -0.05,
            keys: HashSet::new(),
            just_pressed: Vec::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            sun_angle: 0.35,
            sun_light_id,
            fog_enabled: true,
            shafts_enabled: true,
            fog_mode: FogMode::Uniform,
            fog_density: 0.015,
            fog_anisotropy: 0.6,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
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
                        state: ks,
                        physical_key: PhysicalKey::Code(key),
                        repeat,
                        ..
                    },
                ..
            } => match ks {
                ElementState::Pressed => {
                    if !repeat {
                        state.just_pressed.push(key);
                    }
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
                    let grabbed = state
                        .window
                        .set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                        .is_ok();
                    if grabbed {
                        state.window.set_cursor_visible(false);
                        state.cursor_grabbed = true;
                    }
                }
            }

            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                let cfg = wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: state.surface_format,
                    width: size.width,
                    height: size.height,
                    present_mode: wgpu::PresentMode::Fifo,
                    alpha_mode: wgpu::CompositeAlphaMode::Auto,
                    view_formats: vec![],
                    desired_maximum_frame_latency: 2,
                    color_space: wgpu::SurfaceColorSpace::Auto,
                };
                state.surface.configure(&state.device, &cfg);
                state.renderer.set_render_size(size.width, size.height);
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

/// Sun direction as a light "ray direction" (pointing into the scene).
///
/// Orbits in the XY plane with a slight z tilt, so low angles rake across the
/// pillar gaps rather than shining down the hall.
fn sun_light_dir(angle: f32) -> [f32; 3] {
    let to_sun = glam::Vec3::new(angle.cos(), angle.sin(), 0.18).normalize();
    [-to_sun.x, -to_sun.y, -to_sun.z]
}

fn print_help() {
    println!("\n── Volumetric fog demo ──────────────────────────────────");
    println!("  WASD/Space/Shift  move      mouse drag  look (click to grab)");
    println!("  Q/E   rotate sun (low = best shafts)   F  fog on/off");
    println!("  G     light shafts on/off              M  uniform/height fog");
    println!("  1/2   density down/up                  3/4  anisotropy down/up");
    println!("  Dense fog pocket sits mid-hall, z = -6 .. 6 — walk through it.");
    println!("─────────────────────────────────────────────────────────\n");
}

impl AppState {
    fn handle_toggles(&mut self) {
        let pressed = std::mem::take(&mut self.just_pressed);
        let mut dirty = false;

        for key in pressed {
            match key {
                KeyCode::KeyF => {
                    self.fog_enabled = !self.fog_enabled;
                    dirty = true;
                }
                KeyCode::KeyG => {
                    self.shafts_enabled = !self.shafts_enabled;
                    dirty = true;
                }
                KeyCode::KeyM => {
                    self.fog_mode = match self.fog_mode {
                        FogMode::Uniform => FogMode::HeightBased,
                        FogMode::HeightBased => FogMode::Uniform,
                    };
                    dirty = true;
                }
                _ => {}
            }
        }

        if dirty {
            println!(
                "fog={} shafts={} mode={:?} density={:.3} g={:.2}",
                self.fog_enabled,
                self.shafts_enabled,
                self.fog_mode,
                self.fog_density,
                self.fog_anisotropy
            );
        }
    }

    fn render(&mut self, dt: f32) {
        const SPEED: f32 = 6.0;
        const LOOK_SENS: f32 = 0.002;
        const SUN_SPEED: f32 = 0.5;

        self.handle_toggles();

        if self.keys.contains(&KeyCode::KeyQ) {
            self.sun_angle -= SUN_SPEED * dt;
        }
        if self.keys.contains(&KeyCode::KeyE) {
            self.sun_angle += SUN_SPEED * dt;
        }
        if self.keys.contains(&KeyCode::Digit1) {
            self.fog_density = (self.fog_density - 0.05 * dt).max(0.0);
        }
        if self.keys.contains(&KeyCode::Digit2) {
            self.fog_density = (self.fog_density + 0.05 * dt).min(1.0);
        }
        if self.keys.contains(&KeyCode::Digit3) {
            self.fog_anisotropy = (self.fog_anisotropy - 0.5 * dt).max(-0.95);
        }
        if self.keys.contains(&KeyCode::Digit4) {
            self.fog_anisotropy = (self.fog_anisotropy + 0.5 * dt).min(0.95);
        }

        self.cam_yaw += self.mouse_delta.0 * LOOK_SENS;
        self.cam_pitch = (self.cam_pitch - self.mouse_delta.1 * LOOK_SENS).clamp(-1.5, 1.5);
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

        let mut camera = Camera::perspective_look_at(
            self.cam_pos,
            self.cam_pos + forward,
            glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4,
            aspect,
            0.1,
            500.0,
        );

        // Base fog. The mid-hall volume blends against these.
        camera.postprocess_settings.fog_enabled = self.fog_enabled;
        camera.postprocess_settings.fog_mode = self.fog_mode;
        camera.postprocess_settings.fog_density = self.fog_density;
        camera.postprocess_settings.fog_color = [0.62, 0.70, 0.85];
        camera.postprocess_settings.fog_scattering_anisotropy = self.fog_anisotropy;
        camera.postprocess_settings.fog_max_distance = 200.0;
        // Height fog sits at floor level and thins upward.
        camera.postprocess_settings.fog_start_distance = 1.5;
        camera.postprocess_settings.fog_height = 0.0;
        camera.postprocess_settings.fog_height_falloff = 0.25;

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) => t,
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => {
                log::warn!("surface acquire failed");
                return;
            }
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut sun = directional_light(sun_light_dir(self.sun_angle), [1.0, 0.9, 0.75], 4.0);
        sun.god_rays_enabled = self.shafts_enabled as u32;
        let _ = self
            .renderer
            .scene_mut()
            .update_light(self.sun_light_id, sun);

        if let Err(e) = self.renderer.render(&camera, &view) {
            log::error!("Render error: {:?}", e);
        }

        self.renderer.queue().present(output);
    }
}
