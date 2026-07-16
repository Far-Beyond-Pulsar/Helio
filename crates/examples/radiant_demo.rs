// Radiant Material System Demo
// Demonstrates all three tiers of the Radiant material system:
//   Tier 1 — Feature flags on the default PBR uber-shader
//   Tier 2 — Hand-authored surface templates (clear coat, SSS, anisotropic, skin)
//   Tier 3 — Full custom template (iridescent thin-film surface)

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use glam::{EulerRot, Mat4, Quat, Vec3};
use helio::{
    required_wgpu_features, required_wgpu_limits, Camera, Renderer,
    RendererConfig, Scene, SceneActor,
};
use libhelio::{
    FLAG_HAS_NORMAL_MAP, MATERIAL_CLASS_ANISOTROPIC, MATERIAL_CLASS_CLEAR_COAT,
    MATERIAL_CLASS_DEFAULT, MATERIAL_CLASS_SKIN, MATERIAL_CLASS_SUBSURFACE,
};
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
    // Materials for per-frame animation
    animated_iri_id: helio::MaterialId,
    sss_material_id: helio::MaterialId,
    aniso_material_id: helio::MaterialId,
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

        // Tier-2 templates (clear_coat, subsurface, anisotropic, skin) are
        // auto-registered at engine startup by RadiantTemplateRegistry::new()
        // with their MATERIAL_CLASS_* IDs.

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

        let make_mat = v3_demo_common::make_material;

        // Tier 1a: Gold metallic (uber-shader, flags only)
        let gold_mat = renderer.scene_mut().insert_material(make_mat(
            [1.0, 0.75, 0.2, 1.0], 0.2, 1.0, [0.0; 3], 0.0,
        ));

        // Tier 1b: Rough red plastic (uber-shader, flags only)
        let plastic_mat = renderer.scene_mut().insert_material(make_mat(
            [0.9, 0.15, 0.1, 1.0], 0.8, 0.0, [0.0; 3], 0.0,
        ));

        // Tier 2: Clear coat — deep blue metallic with clear-coat template
        // class_params: x=coat_strength, y=coat_roughness, z=coat_IOR, w=unused
        let coat_mat = renderer.scene_mut().insert_material(make_mat(
            [0.05, 0.1, 0.6, 1.0], 0.25, 0.6, [0.0; 3], 0.0,
        ));
        renderer.scene_mut().set_material_class(
            coat_mat, MATERIAL_CLASS_CLEAR_COAT, 0, None,
        ).unwrap();
        renderer.scene_mut().update_material_class_params(
            coat_mat, [1.0, 0.04, 0.0, 0.0], // full strength, very smooth coat
        );

        // Tier 2: PBR + graph snippet override (animated emissive pulse)
        let pulse_mat = renderer.scene_mut().insert_material(make_mat(
            [0.3, 0.3, 0.35, 1.0], 0.5, 0.5, [0.0; 3], 0.0,
        ));
        renderer.scene_mut().set_material_class(
            pulse_mat, MATERIAL_CLASS_DEFAULT, pulse_hash,
            Some(FLAG_HAS_NORMAL_MAP),
        ).unwrap();

        // Tier 2: Opal (subsurface + internal colour flashes)
        // class_params: xyz=subsurface_color (play-of-color tint), w=subsurface_radius (mm)
        // Opal's internal colour is approximated via SSS with a bright, rainbow-tinted
        // transmission colour and very low roughness so light penetrates and scatters
        // within the gemstone body before exiting.
        let sss_mat = renderer.scene_mut().insert_material(make_mat(
            [0.92, 0.88, 0.82, 1.0], 0.04, 0.0, [0.0; 3], 0.0, // milky white body
        ));
        renderer.scene_mut().set_material_class(
            sss_mat, MATERIAL_CLASS_SUBSURFACE, 0, None,
        ).unwrap();
        renderer.scene_mut().update_material_class_params(
            sss_mat, [0.6, 0.8, 0.9, 3.0], // cyan-blue internal tint, 3mm scatter radius
        );

        // Tier 2: Brushed metal (anisotropic GGX)
        // class_params: x=anisotropy (0-1), y=rotation (radians), zw=unused
        let aniso_mat = renderer.scene_mut().insert_material(make_mat(
            [0.85, 0.7, 0.5, 1.0], 0.25, 1.0, [0.0; 3], 0.0, // brass colour
        ));
        renderer.scene_mut().set_material_class(
            aniso_mat, MATERIAL_CLASS_ANISOTROPIC, 0, None,
        ).unwrap();
        renderer.scene_mut().update_material_class_params(
            aniso_mat, [0.85, 0.3, 0.0, 0.0], // strong anisotropy, slight rotation
        );

        // Tier 2: Skin (SSS + dielectric specular at 0.028)
        // class_params: xyz=blood colour, w=subsurface_radius (mm)
        let skin_mat = renderer.scene_mut().insert_material(make_mat(
            [0.85, 0.65, 0.55, 1.0], 0.4, 0.0, [0.0; 3], 0.0, // skin tone
        ));
        renderer.scene_mut().set_material_class(
            skin_mat, MATERIAL_CLASS_SKIN, 0, None,
        ).unwrap();
        renderer.scene_mut().update_material_class_params(
            skin_mat, [0.8, 0.15, 0.05, 3.0], // reddish blood colour, 3mm radius
        );

        // Tier 3: Iridescent thin-film surface (full custom template)
        let iri_mat = renderer.scene_mut().insert_material(make_mat(
            [0.6, 0.6, 0.8, 1.0], 0.15, 0.8, [0.0; 3], 0.0,
        ));
        renderer.scene_mut().set_material_class(
            iri_mat, iridescent_class, 0, None,
        ).unwrap();

        // Animated iridescent (class_params change per-frame)
        let anim_iri_mat = renderer.scene_mut().insert_material(make_mat(
            [0.5, 0.5, 0.6, 1.0], 0.2, 0.6, [0.0; 3], 0.0,
        ));
        renderer.scene_mut().set_material_class(
            anim_iri_mat, iridescent_class, 0, None,
        ).unwrap();

        // ── Meshes ───────────────────────────────────────────────────────────

        let sphere_mesh = renderer.scene_mut().insert_actor(
            SceneActor::mesh(v3_demo_common::sphere_mesh([0.0; 3], 1.0))
        ).as_mesh().unwrap();

        let plane_mesh = renderer.scene_mut().insert_actor(
            SceneActor::mesh(v3_demo_common::plane_mesh([0.0; 3], 12.0))
        ).as_mesh().unwrap();

        // ── Scene objects ────────────────────────────────────────────────────

        let s = 2.7; // spacing
        let yp = 0.0;
        let front_z = 0.0;
        let mid_z = -3.8;
        let back_z = -6.8;

        // Ground plane
        let plane_mat = renderer.scene_mut().insert_material(make_mat(
            [0.10, 0.10, 0.12, 1.0], 0.9, 0.0, [0.0; 3], 0.0,
        ));
        v3_demo_common::insert_object(&mut renderer, plane_mesh, plane_mat,
            Mat4::from_translation(Vec3::new(0.0, -1.5, 0.0)), 12.0);

        // ── Front row: Tier 1 & 2 basics ─────────────────────────────────────

        v3_demo_common::insert_object(&mut renderer, sphere_mesh, gold_mat,
            Mat4::from_translation(Vec3::new(-s * 1.5, yp, front_z)), 1.0);

        v3_demo_common::insert_object(&mut renderer, sphere_mesh, plastic_mat,
            Mat4::from_translation(Vec3::new(-s * 0.5, yp, front_z)), 1.0);

        // Clear coat template (Tier 2)
        v3_demo_common::insert_object(&mut renderer, sphere_mesh, coat_mat,
            Mat4::from_translation(Vec3::new(s * 0.5, yp, front_z)), 1.0);

        // PBR + graph snippet (Tier 2)
        v3_demo_common::insert_object(&mut renderer, sphere_mesh, pulse_mat,
            Mat4::from_translation(Vec3::new(s * 1.5, yp, front_z)), 1.0);

        // ── Middle row: Tier 3 custom ────────────────────────────────────────

        v3_demo_common::insert_object(&mut renderer, sphere_mesh, iri_mat,
            Mat4::from_translation(Vec3::new(-s * 0.5, yp, mid_z)), 1.0);

        v3_demo_common::insert_object(&mut renderer, sphere_mesh, anim_iri_mat,
            Mat4::from_translation(Vec3::new(s * 0.5, yp, mid_z)), 1.0);

        // ── Back row: Tier 2 template demos ──────────────────────────────────

        // Opal (subsurface scattering)
        v3_demo_common::insert_object(&mut renderer, sphere_mesh, sss_mat,
            Mat4::from_translation(Vec3::new(-s * 1.5, yp, back_z)), 1.0);

        // Brushed metal (anisotropic GGX)
        v3_demo_common::insert_object(&mut renderer, sphere_mesh, aniso_mat,
            Mat4::from_translation(Vec3::new(-s * 0.5, yp, back_z)), 1.0);

        // Skin
        v3_demo_common::insert_object(&mut renderer, sphere_mesh, skin_mat,
            Mat4::from_translation(Vec3::new(s * 0.5, yp, back_z)), 1.0);

        // Anisotropic with rotating direction (animated)
        let aniso2_mat = renderer.scene_mut().insert_material(make_mat(
            [0.6, 0.6, 0.7, 1.0], 0.15, 0.9, [0.0; 3], 0.0,
        ));
        renderer.scene_mut().set_material_class(
            aniso2_mat, MATERIAL_CLASS_ANISOTROPIC, 0, None,
        ).unwrap();
        v3_demo_common::insert_object(&mut renderer, sphere_mesh, aniso2_mat,
            Mat4::from_translation(Vec3::new(s * 1.5, yp, back_z)), 1.0);

        // ── Lights ───────────────────────────────────────────────────────────

        let sun_id = renderer.scene_mut().insert_actor(SceneActor::light(
            v3_demo_common::directional_light(
                [0.4, -0.75, 0.3], [1.0, 0.95, 0.85], 3.5,
            ),
        )).as_light().unwrap();

        renderer.scene_mut().insert_actor(SceneActor::light(
            v3_demo_common::directional_light(
                [-0.3, -0.4, -0.5], [0.5, 0.6, 0.8], 0.6,
            ),
        ));

        renderer.scene_mut().insert_actor(SceneActor::light(
            v3_demo_common::directional_light(
                [0.0, 0.5, -0.8], [0.3, 0.4, 0.6], 0.4,
            ),
        ));

        // ── Sky ──────────────────────────────────────────────────────────────

        renderer.scene_mut().insert_actor(SceneActor::Sky(
            helio::SkyActor::new().with_sky_color([0.12, 0.15, 0.28]),
        ));
        renderer.set_ambient([0.04, 0.04, 0.08], 0.08);

        // ── Print legend ─────────────────────────────────────────────────────

        log::info!("");
        log::info!("═══ Helio Radiant Material Demo ═══");
        log::info!("  ── Front row ──");
        log::info!("    Gold metallic      (Tier 1, uber-shader, flags only)");
        log::info!("    Red plastic        (Tier 1, uber-shader, flags only)");
        log::info!("    Clear coat         (Tier 2, clear_coat template)");
        log::info!("    Emissive pulse     (Tier 2, PBR + graph snippet)");
        log::info!("  ── Middle row ──");
        log::info!("    Iridescent static  (Tier 3, custom template)");
        log::info!("    Iridescent anim    (Tier 3, animated class_params)");
        log::info!("  ── Back row ──");
        log::info!("    Opal (play-of-colour)  (Tier 2, subsurface template, animated)");
        log::info!("    Brushed metal          (Tier 2, anisotropic template)");
        log::info!("    Skin                   (Tier 2, skin template)");
        log::info!("    Brushed metal (moving) (Tier 2, anisotropic, anim direction)");
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
            cam_pos: Vec3::new(0.0, 2.0, 10.0),
            yaw: 0.0,
            pitch: -0.12,
            velocity: Vec3::ZERO,
            keys: HashSet::new(),
            cursor_grabbed: false,
            mouse_delta: (0.0, 0.0),
            animated_iri_id: anim_iri_mat,
            sss_material_id: sss_mat,
            aniso_material_id: aniso2_mat,
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

                // Slow sun orbit
                let elapsed = now.duration_since(state.last_frame).as_secs_f32();
                let angle = elapsed * 0.015 + state.yaw;
                let sun_pos = Vec3::new(angle.sin() * 14.0, 7.0, angle.cos() * 14.0);
                let _ = state.renderer.scene_mut().update_light(
                    state.sun_light_id,
                    v3_demo_common::directional_light(
                        [-sun_pos.x, -sun_pos.y, -sun_pos.z], [1.0, 0.95, 0.85], 3.5,
                    ),
                );

                // Animate iridescent params
                let t = now.duration_since(Instant::now()).as_secs_f32();
                state.renderer.scene_mut().update_material_class_params(
                    state.animated_iri_id,
                    [3.0 + (t * 0.3).sin() * 2.0, 0.5 + (t * 0.5).sin() * 0.5, 0.0, 0.0],
                );

                // Animate opal: cycle internal colour through the spectrum (play-of-colour)
                let hue = (t * 0.3).sin() * 0.5 + 0.5;
                let opal_color = [
                    0.5 + (t * 1.1).cos() * 0.4,
                    0.5 + (t * 1.3 + 2.0).cos() * 0.4,
                    0.5 + (t * 0.9 + 4.0).cos() * 0.4,
                ];
                let sss_radius = 2.5 + (t * 0.5).sin() * 0.8;
                state.renderer.scene_mut().update_material_class_params(
                    state.sss_material_id,
                    [opal_color[0], opal_color[1], opal_color[2], sss_radius],
                );

                // Animate anisotropic: rotate the brush direction
                let aniso_rot = t * 0.4;
                state.renderer.scene_mut().update_material_class_params(
                    state.aniso_material_id,
                    [0.8, aniso_rot, 0.0, 0.0],
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
