// Radiant Material System Demo
// Demonstrates all three tiers of the Radiant material system:
//   Tier 1 — Feature flags on the default PBR uber-shader
//   Tier 2 — Template + graph-snippet override
//   Tier 3 — Full custom template (iridescent thin-film surface)

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use glam::{EulerRot, Mat4, Quat, Vec3};
use helio::{
    required_wgpu_features, required_wgpu_limits, Camera, Renderer,
    RendererConfig, Scene, SceneActor,
};
use libhelio::{FLAG_HAS_NORMAL_MAP, MATERIAL_CLASS_DEFAULT};
use helio_default_graphs::build_default_graph;
use helio_pass_gbuffer::GBufferPass;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalPosition,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

mod v3_demo_common;

const LOOK_SENS: f32 = 0.002;
const FLY_SPEED: f32 = 10.0;
const DRAG: f32 = 6.0;

// ── Graph snippet for Tier 2 — animated emissive pulse ──────────────────────

const GRAPH_EMISSIVE_PULSE: &str = "\
{
    let t = f32(globals.frame) * 0.05;
    let pulse = sin(t) * 0.5 + 0.5;
    let pulse_color = vec3<f32>(1.0, 0.3, 0.1) * pulse * 2.0;
    emissive = emissive + pulse_color;
}
";

// ── App ──────────────────────────────────────────────────────────────────────

struct App {
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
    alpha_mode: wgpu::CompositeAlphaMode,
    renderer: Renderer,
    last_frame: Instant,
    cam_pos: Vec3,
    yaw: f32,
    pitch: f32,
    velocity: Vec3,
    keys: HashSet<KeyCode>,
    cursor_grabbed: bool,
    mouse_delta: (f32, f32),
    // Scene objects for per-frame manipulation
    animated_mat_id: helio::MaterialId,
    sun_light_id: helio::LightId,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() { return; }

        let attrs = Window::default_attributes()
            .with_title("Helio Radiant Material Demo")
            .with_inner_size(winit::dpi::PhysicalSize::new(1600, 900));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("No suitable GPU adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: required_wgpu_features(adapter.features()),
                required_limits: required_wgpu_limits(adapter.limits()),
                ..Default::default()
            },
        ))
        .expect("Device request failed");

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        device.on_uncaptured_error(Arc::new(|error| {
            log::error!("wgpu uncaptured error: {}", error);
        }));

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let alpha_mode = caps.alpha_modes[0];

        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
                color_space: wgpu::SurfaceColorSpace::Auto,
            },
        );

        // ── Renderer setup ──────────────────────────────────────────────────

        let config = RendererConfig::new(size.width, size.height, surface_format);
        let mut scene = Scene::new(device.clone(), queue.clone());
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
        let debug_state = Arc::new(std::sync::Mutex::new(helio::DebugDrawState::default()));

        let graph = build_default_graph(
            &device, &queue, &scene, config,
            debug_state.clone(), &debug_camera_buf, &cull_stats_buf, None,
        );

        let mut renderer = Renderer::new(
            device.clone(), queue.clone(),
            config.surface_format, config.width, config.height, config.render_scale,
            config, scene, graph, debug_state, debug_camera_buf, cull_stats_buf,
        );

        // ── Register the iridescent template (Tier 3) ────────────────────────

        let iridescent_wgsl = include_str!("shaders/radiant_iridescent.wgsl");
        let iridescent_class = renderer
            .find_pass_mut::<GBufferPass>()
            .expect("GBufferPass not found in graph")
            .template_registry_mut()
            .register_str("iridescent", iridescent_wgsl.to_string());

        log::info!("[RADIANT] Iridescent template registered as class {}", iridescent_class);

        // ── Register a graph snippet (Tier 2) ────────────────────────────────

        let pulse_hash = 0xA3F10001u64;
        renderer.scene_mut().radiant_graphs.register(pulse_hash, GRAPH_EMISSIVE_PULSE.to_string());

        // ── Materials ────────────────────────────────────────────────────────

        // Tier 1a: Gold metallic (uber-shader, flags only)
        let gold_mat = renderer.scene_mut().insert_material(v3_demo_common::make_material(
            [1.0, 0.75, 0.2, 1.0], 0.2, 1.0, [0.0; 3], 0.0,
        ));

        // Tier 1b: Rough red plastic (uber-shader, flags only)
        let plastic_mat = renderer.scene_mut().insert_material(v3_demo_common::make_material(
            [0.9, 0.15, 0.1, 1.0], 0.8, 0.0, [0.0; 3], 0.0,
        ));

        // Tier 1c: Blue dielectric with clear-coat flag (just a flag toggle, no PSO switch)
        let clear_coat_mat = renderer.scene_mut().insert_material(v3_demo_common::make_material(
            [0.15, 0.3, 0.85, 1.0], 0.3, 0.0, [0.0; 3], 0.0,
        ));

        // Tier 2: PBR + graph snippet override (animated emissive pulse)
        let pulse_mat = renderer.scene_mut().insert_material(v3_demo_common::make_material(
            [0.3, 0.3, 0.35, 1.0], 0.5, 0.5, [0.0; 3], 0.0,
        ));
        renderer.scene_mut().set_material_class(
            pulse_mat, MATERIAL_CLASS_DEFAULT, pulse_hash,
            Some(FLAG_HAS_NORMAL_MAP),
        ).unwrap();

        // Tier 3: Iridescent thin-film surface (full custom template)
        let iri_mat = renderer.scene_mut().insert_material(v3_demo_common::make_material(
            [0.6, 0.6, 0.8, 1.0], 0.15, 0.8, [0.0; 3], 0.0,
        ));
        renderer.scene_mut().set_material_class(
            iri_mat, iridescent_class, 0, None,
        ).unwrap();

        // ── Animated material (class_params changes per frame) ───────────────
        let anim_mat = renderer.scene_mut().insert_material(v3_demo_common::make_material(
            [0.5, 0.5, 0.6, 1.0], 0.2, 0.6, [0.0; 3], 0.0,
        ));
        renderer.scene_mut().set_material_class(
            anim_mat, iridescent_class, 0, None,
        ).unwrap();

        // ── Meshes ───────────────────────────────────────────────────────────

        let sphere_mesh = renderer.scene_mut().insert_actor(
            SceneActor::mesh(v3_demo_common::sphere_mesh([0.0; 3], 1.0))
        ).as_mesh().unwrap();

        let plane_mesh = renderer.scene_mut().insert_actor(
            SceneActor::mesh(v3_demo_common::plane_mesh([0.0; 3], 10.0))
        ).as_mesh().unwrap();

        // ── Scene objects ────────────────────────────────────────────────────

        let spacing = 2.8;
        let y_pos = 0.0;

        // Ground plane (simple, no flags)
        let plane_mat = renderer.scene_mut().insert_material(v3_demo_common::make_material(
            [0.12, 0.12, 0.14, 1.0], 0.9, 0.0, [0.0; 3], 0.0,
        ));
        v3_demo_common::insert_object(&mut renderer, plane_mesh, plane_mat,
            Mat4::from_translation(Vec3::new(0.0, -1.5, 0.0)), 14.0);

        // Tier 1: Gold metallic
        v3_demo_common::insert_object(&mut renderer, sphere_mesh, gold_mat,
            Mat4::from_translation(Vec3::new(-spacing * 1.5, y_pos, 0.0)), 1.0);

        // Tier 1: Red plastic
        v3_demo_common::insert_object(&mut renderer, sphere_mesh, plastic_mat,
            Mat4::from_translation(Vec3::new(-spacing * 0.5, y_pos, 0.0)), 1.0);

        // Tier 1: Blue clear-coat
        v3_demo_common::insert_object(&mut renderer, sphere_mesh, clear_coat_mat,
            Mat4::from_translation(Vec3::new(spacing * 0.5, y_pos, 0.0)), 1.0);

        // Tier 2: PBR + emissive pulse graph
        v3_demo_common::insert_object(&mut renderer, sphere_mesh, pulse_mat,
            Mat4::from_translation(Vec3::new(spacing * 1.5, y_pos, 0.0)), 1.0);

        // Tier 3: Iridescent (static params)
        v3_demo_common::insert_object(&mut renderer, sphere_mesh, iri_mat,
            Mat4::from_translation(Vec3::new(-spacing * 1.0, y_pos, -3.5)), 1.0);

        // Tier 3: Iridescent (animated params — class_params.x/y change per frame)
        v3_demo_common::insert_object(&mut renderer, sphere_mesh, anim_mat,
            Mat4::from_translation(Vec3::new(spacing * 1.0, y_pos, -3.5)), 1.0);

        // ── Lights ───────────────────────────────────────────────────────────

        // Sun
        let sun_id = renderer.scene_mut().insert_actor(SceneActor::light(v3_demo_common::directional_light(
            [0.4, -0.75, 0.3], [1.0, 0.95, 0.85], 3.0,
        ))).as_light().unwrap();

        // Fill
        renderer.scene_mut().insert_actor(SceneActor::light(v3_demo_common::directional_light(
            [-0.3, -0.4, -0.5], [0.5, 0.6, 0.8], 0.6,
        )));

        // Rim
        renderer.scene_mut().insert_actor(SceneActor::light(v3_demo_common::directional_light(
            [0.0, 0.5, -0.8], [0.3, 0.4, 0.6], 0.4,
        )));

        // ── Sky ──────────────────────────────────────────────────────────────

        renderer.scene_mut().insert_actor(SceneActor::Sky(
            helio::SkyActor::new().with_sky_color([0.15, 0.2, 0.35]),
        ));
        renderer.set_ambient([0.05, 0.05, 0.1], 0.06);

        // ── Print legend ─────────────────────────────────────────────────────

        log::info!("");
        log::info!("═══ Helio Radiant Material Demo ═══");
        log::info!("  Tier 1 — Uber-shader (flags only):");
        log::info!("    Leftmost:   Gold metallic  (FLAG_HAS_NORMAL_MAP)");
        log::info!("    Center-L:   Red plastic    (uber, no flags)");
        log::info!("    Center-R:   Blue clear-coat (FLAG_HAS_CLEAR_COAT)");
        log::info!("  Tier 2 — PBR + graph snippet:");
        log::info!("    Rightmost:  Animated emissive pulse via graph override");
        log::info!("  Tier 3 — Custom template:");
        log::info!("    Back-left:  Iridescent thin-film (static params)");
        log::info!("    Back-right: Iridescent thin-film (animated params)");
        log::info!("");
        log::info!("  Controls: WASD fly, mouse look, Space/Shift up/down");
        log::info!("");

        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_format,
            alpha_mode,
            renderer,
            last_frame: Instant::now(),
            cam_pos: Vec3::new(0.0, 2.5, 9.0),
            yaw: 0.0,
            pitch: -0.15,
            velocity: Vec3::ZERO,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            animated_mat_id: anim_mat,
            sun_light_id: sun_id,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    state.surface.configure(
                        &state.device,
                        &wgpu::SurfaceConfiguration {
                            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                            format: state.surface_format,
                            width: new_size.width,
                            height: new_size.height,
                            present_mode: wgpu::PresentMode::Fifo,
                            alpha_mode: state.alpha_mode,
                            view_formats: vec![],
                            desired_maximum_frame_latency: 2,
                            color_space: wgpu::SurfaceColorSpace::Auto,
                        },
                    );
                    state.renderer.set_render_size(new_size.width, new_size.height);
                }
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(KeyCode::Escape),
                    ..
                },
                ..
            } => {
                if state.cursor_grabbed {
                    state.cursor_grabbed = false;
                    state.window.set_cursor_visible(true);
                    let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                } else {
                    event_loop.exit();
                }
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(code),
                    ..
                },
                ..
            } => { let _ = state.keys.insert(code); }
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state: ElementState::Released,
                    physical_key: PhysicalKey::Code(code),
                    ..
                },
                ..
            } => { state.keys.remove(&code); }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } if !state.cursor_grabbed => {
                let ok = state.window
                    .set_cursor_grab(CursorGrabMode::Confined)
                    .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                    .is_ok();
                if ok {
                    state.cursor_grabbed = true;
                    state.window.set_cursor_visible(false);
                }
            }
            WindowEvent::CursorMoved {
                position: pos,
                ..
            } if state.cursor_grabbed => {
                let center = (
                    state.window.inner_size().width as f64 / 2.0,
                    state.window.inner_size().height as f64 / 2.0,
                );
                state.mouse_delta.0 += (pos.x - center.0) as f32;
                state.mouse_delta.1 += (pos.y - center.1) as f32;
                let _ = state.window.set_cursor_position(
                    PhysicalPosition::new(center.0 as i32, center.1 as i32)
                );
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(state.last_frame).as_secs_f32().min(0.05);
                state.last_frame = now;
                state.update(dt);
                let size = state.window.inner_size();
                let camera = state.camera(size.width, size.height);

                // Update sun position with a slow orbit
                let angle = now.duration_since(state.last_frame).as_secs_f32() * 0.02 + state.yaw;
                let sun_pos = glam::Vec3::new(angle.sin() * 12.0, 6.0, angle.cos() * 12.0);
                let _ = state.renderer.scene_mut().update_light(
                    state.sun_light_id,
                    v3_demo_common::directional_light(
                        [-sun_pos.x, -sun_pos.y, -sun_pos.z], [1.0, 0.95, 0.85], 3.0,
                    ),
                );

                // Animate class_params on the animated iridescent sphere
                let t = now.duration_since(Instant::now()).as_secs_f32();
                let freq = 3.0 + (t * 0.3).sin() * 2.0;
                let intensity = 0.5 + (t * 0.5).sin() * 0.5;
                state.renderer.scene_mut().update_material_class_params(
                    state.animated_mat_id, [freq, intensity, 0.0, 0.0],
                );

                let output = match state.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(t)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
                    _ => return,
                };
                let view = output.texture.create_view(
                    &wgpu::TextureViewDescriptor::default()
                );
                if let Err(e) = state.renderer.render(&camera, &view) {
                    log::error!("render error: {:?}", e);
                }
                state.queue.present(output);
                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        let Some(state) = &mut self.state else { return };
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if state.cursor_grabbed {
                state.mouse_delta.0 += dx as f32;
                state.mouse_delta.1 += dy as f32;
            }
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

impl AppState {
    fn update(&mut self, dt: f32) {
        let (dx, dy) = self.mouse_delta;
        self.mouse_delta = (0.0, 0.0);
        self.yaw -= dx * LOOK_SENS;
        self.pitch = (self.pitch - dy * LOOK_SENS).clamp(-1.5, 1.5);

        let orientation = Quat::from_euler(EulerRot::YXZ, self.yaw, self.pitch, 0.0);
        let forward = orientation * -Vec3::Z;
        let right = orientation * Vec3::X;

        let mut accel = Vec3::ZERO;
        if self.keys.contains(&KeyCode::KeyW) { accel += forward; }
        if self.keys.contains(&KeyCode::KeyS) { accel -= forward; }
        if self.keys.contains(&KeyCode::KeyA) { accel -= right; }
        if self.keys.contains(&KeyCode::KeyD) { accel += right; }
        if self.keys.contains(&KeyCode::Space) { accel += Vec3::Y; }
        if self.keys.contains(&KeyCode::ShiftLeft) { accel -= Vec3::Y; }

        self.velocity += accel * FLY_SPEED * dt;
        self.velocity /= 1.0 + DRAG * dt;
        self.cam_pos += self.velocity * dt;
    }

    fn camera(&self, width: u32, height: u32) -> Camera {
        let orientation = Quat::from_euler(EulerRot::YXZ, self.yaw, self.pitch, 0.0);
        let target = self.cam_pos + orientation * -Vec3::Z;
        let up = orientation * Vec3::Y;
        Camera::perspective_look_at(
            self.cam_pos,
            target,
            up,
            std::f32::consts::FRAC_PI_4,
            width as f32 / height.max(1) as f32,
            0.01,
            2000.0,
        )
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App { state: None };
    event_loop.run_app(&mut app).unwrap();
}
